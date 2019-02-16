#include <opencv2/core/core.hpp>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/layers/image_label_data_layer.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include <opencv2/imgproc/imgproc.hpp>

namespace {

cv::Mat PadImage(cv::Mat &image, int min_size, double value = -1) {
  if (image.rows >= min_size && image.cols >= min_size) {
    return image;
  }
  int top, bottom, left, right;
  top = bottom = left = right = 0;
  if (image.rows < min_size) {
    top = (min_size - image.rows) / 2;
    bottom = min_size - image.rows - top;
  }

  if (image.cols < min_size) {
    left = (min_size - image.cols) / 2;
    right = min_size - image.cols - left;
  }
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, top, bottom, right, left,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

cv::Mat ExtendLabelMargin(cv::Mat &image, int margin_w, int margin_h,
                          double value = -1) {
  cv::Mat big_image;
  if (value < 0) {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_REFLECT_101);
  } else {
    cv::copyMakeBorder(image, big_image, margin_h, margin_h, margin_w, margin_w,
                       cv::BORDER_CONSTANT, cv::Scalar(value));
  }
  return big_image;
}

template <typename Dtype>
void GetLabelSlice(const Dtype *labels, int rows, int cols,
                   const caffe::Slice &label_slice, Dtype *slice_data) {
  // for (int c = 0; c < channels; ++c) {
  labels += label_slice.offset(0) * cols;
  for (int h = 0; h < label_slice.dim(0); ++h) {
    labels += label_slice.offset(1);
    for (int w = 0; w < label_slice.dim(1); ++w) {
      slice_data[w] = labels[w * label_slice.stride(1)];
    }
    labels += cols * label_slice.stride(0) - label_slice.offset(1);
    slice_data += label_slice.dim(1);
  }
  //t_label_data += this->label_margin_h_ * (label_width + this->label_margin_w_ * 2);
  // }
}

}

namespace caffe {

template <typename Dtype>
ImageLabelDataLayer<Dtype>::ImageLabelDataLayer(
    const LayerParameter &param) : BasePrefetchingDataLayer<Dtype>(param) {
  std::random_device rand_dev;
  rng_ = new std::mt19937(rand_dev());
}

template <typename Dtype>
ImageLabelDataLayer<Dtype>::~ImageLabelDataLayer() {
  this->StopInternalThread();
  delete rng_;
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                const vector<Blob<Dtype>*>& top) {
  auto &data_param = this->layer_param_.image_label_data_param();
  string data_dir = data_param.data_dir();
  string image_dir = data_param.image_dir();
  string label_dir = data_param.label_dir();

  if (image_dir == "" && label_dir == "" && data_dir != "") {
    image_dir = data_dir;
    label_dir = data_dir;
  }

  // Read the file with filenames and labels
  const string& image_list_path =
      this->layer_param_.image_label_data_param().image_list_path();
  LOG(INFO) << "Opening image list " << image_list_path;
  std::ifstream infile(image_list_path.c_str());
  string filename;
  while (infile >> filename) {
    image_lines_.push_back(filename);
  }

  const string& label_list_path =
      this->layer_param_.image_label_data_param().label_list_path();
  LOG(INFO) << "Opening label list " << image_list_path;
  std::ifstream in_label(label_list_path.c_str());
  while (in_label >> filename) {
    label_lines_.push_back(filename);
  }

  if (this->layer_param_.image_label_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << image_lines_.size() << " images.";
  LOG(INFO) << "A total of " << label_lines_.size() << " label.";
  CHECK_EQ(image_lines_.size(), label_lines_.size());

  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_label_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
                        this->layer_param_.image_label_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(image_lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_]);
  CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }
  cv_img = PadImage(cv_img, crop_size);

  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> data_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(data_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_label_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  data_shape[0] = batch_size;
  top[0]->Reshape(data_shape);

  /*
   * label
   */
  auto &label_slice = this->layer_param_.image_label_data_param().label_slice();
  label_margin_h_ = label_slice.offset(0);
  label_margin_w_ = label_slice.offset(1);
  LOG(INFO) << "Assuming image and label map sizes are the same";
  vector<int> label_shape(4);
  label_shape[0] = batch_size;
  label_shape[1] = 1;
  label_shape[2] = label_slice.dim(0);
  label_shape[3] = label_slice.dim(1);
  top[1]->Reshape(label_shape);

  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(data_shape);
    this->prefetch_[i]->label_.Reshape(label_shape);
  }

  LOG(INFO) << "output data size: " << top[0]->num() << ","
  << top[0]->channels() << "," << top[0]->height() << ","
  << top[0]->width();

  LOG(INFO) << "output label size: " << top[1]->num() << ","
  << top[1]->channels() << "," << top[1]->height() << ","
  << top[1]->width();
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::ShuffleImages() {
//  caffe::rng_t* prefetch_rng =
//      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
//  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
//  LOG(FATAL) <<
//      "ImageLabelDataLayer<Dtype>::ShuffleImages() is not implemented";
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  vector<int> order(image_lines_.size());
  for (int i = 0; i < order.size(); ++i) {
    order[i] = i;
  }
  shuffle(order.begin(), order.end(), prefetch_rng);
  vector<std::string> new_image_lines(image_lines_.size());
  vector<std::string> new_label_lines(label_lines_.size());
  for (int i = 0; i < order.size(); ++i) {
    new_image_lines[i] = image_lines_[order[i]];
    new_label_lines[i] = label_lines_[order[i]];
  }
  swap(image_lines_, new_image_lines);
  swap(label_lines_, new_label_lines);
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::SampleScale(cv::Mat *image, cv::Mat *label) {
  ImageLabelDataParameter data_param =
      this->layer_param_.image_label_data_param();
  if (!data_param.rand_scale()) return;
  double scale = std::uniform_real_distribution<double>(
      data_param.min_scale(), data_param.max_scale())(*rng_);
  cv::Size zero_size(0, 0);
  cv::resize(*label, *label, cv::Size(0, 0),
             scale, scale, cv::INTER_NEAREST);

  if (scale > 1) {
    cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_CUBIC);
  } else {
    cv::resize(*image, *image, zero_size, scale, scale, cv::INTER_AREA);
  }
}

template <typename Dtype>
void AssignEvenLabelWeight(const Dtype *labels, int num, Dtype *weights) {
  Dtype max_label = labels[0];
  for (int i = 0; i < num; ++i) {
    if (labels[i] != 255) {
      max_label = std::max(labels[i], max_label);
    }
  }
  int num_labels = static_cast<int>(max_label) + 1;
  vector<int> counts(num_labels, 0);
  vector<double> label_weight(num_labels);
  for (int i = 0; i < num; ++i) {
    if (labels[i] != 255) {
      counts[static_cast<int>(labels[i])] += 1;
    }
  }
  for (int i = 0; i < num_labels; ++i) {
    if (counts[i] == 0) {
      label_weight[i] = 0;
    } else {
      label_weight[i] = 1.0 / counts[i];
    }
  }
  for (int i = 0; i < num; ++i) {
    weights[i] = label_weight[static_cast<int>(labels[i])];
  }
}

template <typename Dtype>
void ImageLabelDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->transformed_data_.count());
  ImageLabelDataParameter data_param =
      this->layer_param_.image_label_data_param();
  const int batch_size = data_param.batch_size();
  string data_dir = data_param.data_dir();
  string image_dir =
      this->layer_param_.image_label_data_param().image_dir();
  string label_dir =
      this->layer_param_.image_label_data_param().label_dir();

  if (image_dir == "" && label_dir == "" && data_dir != "") {
    image_dir = data_dir;
    label_dir = data_dir;
  }

  int crop_size = -1;
  auto transform_param = this->layer_param_.transform_param();
  if (transform_param.has_crop_size()) {
    crop_size = transform_param.crop_size();
  }

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_], true);
  cv_img = PadImage(cv_img, crop_size);

  CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);


  cv::Mat cv_label = ReadImageToCVMat(label_dir + label_lines_[lines_id_],
                                      false);
  cv_label = PadImage(cv_label, crop_size);

  CHECK(cv_label.data) << "Could not load " << label_lines_[lines_id_];
  vector<int> label_shape = this->data_transformer_->InferBlobShape(cv_label);

  this->transformed_label_.Reshape(label_shape);

  auto &label_slice = this->layer_param_.image_label_data_param().label_slice();

  label_shape[0] = batch_size;
  label_shape[2] = label_slice.dim(0);
  label_shape[3] = label_slice.dim(1);
  batch->label_.Reshape(label_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  auto lines_size = image_lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(image_dir + image_lines_[lines_id_]);
    cv::Mat cv_label = ReadImageToCVMat(label_dir + label_lines_[lines_id_],
                                        false);
    SampleScale(&cv_img, &cv_label);
    switch (data_param.padding()) {
      case ImageLabelDataParameter_Padding_ZERO:
        cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, 0);
        cv_img = PadImage(cv_img, crop_size, 0);
        break;
      case ImageLabelDataParameter_Padding_REFLECT:
        cv_img = ExtendLabelMargin(cv_img, label_margin_w_, label_margin_h_, -1);
        cv_img = PadImage(cv_img, crop_size, -1);
        break;
      default:
        LOG(FATAL) << "Unknown Padding";
    }
    cv_label = ExtendLabelMargin(cv_label, label_margin_w_, label_margin_h_, 255);
    cv_label = PadImage(cv_label, crop_size, 255);

    CHECK(cv_img.data) << "Could not load " << image_lines_[lines_id_];
    CHECK(cv_label.data) << "Could not load " << label_lines_[lines_id_];
    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image

    int image_offset = batch->data_.offset(item_id);
    int label_offset = batch->label_.offset(item_id);
    this->transformed_data_.set_cpu_data(prefetch_data + image_offset);
    // this->transformed_label_.set_cpu_data(prefetch_label + label_offset);
    this->data_transformer_->Transform(cv_img, cv_label,
                                       &(this->transformed_data_),
                                       &(this->transformed_label_));

    Dtype *label_data = prefetch_label + label_offset;
    const Dtype *t_label_data = this->transformed_label_.cpu_data();
//    for (int c = 0; c < label_channel; ++c) {
//      t_label_data += this->label_margin_h_ * (label_width + this->label_margin_w_ * 2);
//      for (int h = 0; h < label_height; ++h) {
//        t_label_data += this->label_margin_w_;
//        for (int w = 0; w < label_width; ++w) {
//          label_data[w] = t_label_data[w];
//        }
//        t_label_data += this->label_margin_w_ + label_width;
//        label_data += label_width;
//      }
//      t_label_data += this->label_margin_h_ * (label_width + this->label_margin_w_ * 2);
//    }
    GetLabelSlice(t_label_data, crop_size, crop_size, label_slice, label_data);
//    CHECK_EQ(t_label_data - this->transformed_label_.cpu_data(),
//             this->transformed_label_.count());
//    cv::Mat_<Dtype> cropped_label(label_height, label_width,
//                                  prefetch_label + label_offset);
//    cropped_label = transformed_label(
//        cv::Range(label_margin_h_, label_margin_h_ + label_height),
//        cv::Range(label_margin_w_, label_margin_w_ + label_width));
    trans_time += timer.MicroSeconds();

    // prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_label_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }

  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageLabelDataLayer);
REGISTER_LAYER_CLASS(ImageLabelData);

}  // namespace caffe
