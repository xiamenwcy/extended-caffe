#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/focal_softmax_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FocalSoftmaxLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  LayerParameter softmax_param(this->layer_param_);
  softmax_param.set_type("Softmax");
  softmax_layer_ = LayerRegistry<Dtype>::CreateLayer(softmax_param);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  gamma_ = this->layer_param_.focal_softmax_loss_param().gamma();
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (!this->layer_param_.loss_param().has_normalization() &&
      this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = this->layer_param_.loss_param().normalization();
  }
  
  weight_by_label_freqs_ =
    this->layer_param_.loss_param().weight_by_label_freqs();
  
  if (weight_by_label_freqs_) {
    vector<int> count_shape(1, this->layer_param_.loss_param().class_weighting_size());
    label_counts_.Reshape(count_shape);
    CHECK_EQ(this->layer_param_.loss_param().class_weighting_size(), bottom[0]->channels())
		<< "Number of class weight values does not match the number of classes.";
    float* label_count_data = label_counts_.mutable_cpu_data();
    for (int i = 0; i < this->layer_param_.loss_param().class_weighting_size(); i++) {
        label_count_data[i] = this->layer_param_.loss_param().class_weighting(i);
    }
  }
  else
  {
	vector<int> count_shape(1, bottom[0]->channels());
    label_counts_.Reshape(count_shape);
    float* label_count_data = label_counts_.mutable_cpu_data();
    for (int i = 0; i < bottom[0]->channels(); i++) {
        label_count_data[i] = 1.0;
    }
  }
  
}

template <typename Dtype>
void FocalSoftmaxLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  softmax_axis_ =
      bottom[0]->CanonicalAxisIndex(this->layer_param_.softmax_param().axis());
 
  outer_num_ = bottom[0]->count(0, softmax_axis_); // n
  inner_num_ = bottom[0]->count(softmax_axis_ + 1);// h * w
  CHECK_EQ(outer_num_ * inner_num_, bottom[1]->count())
      << "Number of labels must match number of predictions; "
      << "e.g., if softmax axis == 1 and prediction shape is (N, C, H, W), "
      << "label count (number of labels) must be N*H*W, "
      << "with integer values in {0, 1, ..., C-1}.";
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
    if (weight_by_label_freqs_) {
    CHECK_EQ(this->layer_param_.loss_param().class_weighting_size(), bottom[0]->channels())
		<< "Number of class weight values does not match the number of classes.";
  }
}
template <typename Dtype>
Dtype FocalSoftmaxLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, Dtype valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
        normalizer = valid_count;
      }
      break;
    case LossParameter_NormalizationMode_BATCH_SIZE:
      normalizer = Dtype(outer_num_);
      break;
    case LossParameter_NormalizationMode_NONE:
      normalizer = Dtype(1);
      break;
    default:
      LOG(FATAL) << "Unknown normalization mode: "
          << LossParameter_NormalizationMode_Name(normalization_mode);
  }
  // Some users will have no labels for some examples in order to 'turn off' a
  // particular loss in a multi-task setup. The max prevents NaNs in that case.
  return std::max(Dtype(1.0), normalizer);
}
template <typename Dtype>
void FocalSoftmaxLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int dim = prob_.count() / outer_num_;
  int count = 0;
  Dtype loss = 0;
  Dtype pt = 0;
  for (int i = 0; i < outer_num_; ++i) {
    for (int j = 0; j < inner_num_; j++) {
      const int label_value = static_cast<int>(label[i * inner_num_ + j]);
      if (has_ignore_label_ && label_value == ignore_label_) {
        continue;
      }
      DCHECK_GE(label_value, 0);
      DCHECK_LT(label_value, prob_.shape(softmax_axis_));

	  const float* label_count_data = label_counts_.cpu_data();
      pt = prob_data[i * dim + label_value * inner_num_ + j];
      loss -=  static_cast<Dtype>(label_count_data[label_value])* pow(1.0 - pt, gamma_) * log(std::max(pt, Dtype(FLT_MIN)));
      ++count;
    }
  }
  normalizer_= get_normalizer( normalization_, count);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void FocalSoftmaxLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    //caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int dim = prob_.count() / outer_num_;
    Dtype focal_diff = 0;
    Dtype pt = 0;
    Dtype pc = 0;
	const float* label_count_data = label_counts_.cpu_data();
    for (int i = 0; i < outer_num_; ++i) {
      for (int j = 0; j < inner_num_; ++j) {
        const int label_value = static_cast<int>(label[i * inner_num_ + j]);
        if (has_ignore_label_ && label_value == ignore_label_) {
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
            bottom_diff[i * dim + c * inner_num_ + j] = 0;
          }
        } else {
          //bottom_diff[i * dim + label_value * inner_num_ + j] -= 1;
          pt = prob_data[i * dim + label_value * inner_num_ + j];
          for (int c = 0; c < bottom[0]->shape(softmax_axis_); ++c) {
              pc = prob_data[i * dim + c * inner_num_ + j];
              if(c == label_value){
                  focal_diff =  pow(1 - pt, gamma_) * (gamma_ * pt * log(std::max(pt, Dtype(FLT_MIN))) + pt - 1);
              }
              else{
                  focal_diff = (pow(1 - pt, gamma_ - 1) * (-gamma_ * log(std::max(pt, Dtype(FLT_MIN))) * pt * pc)
                       + pow(1 - pt, gamma_) * pc);
              }
              bottom_diff[i * dim + c * inner_num_ + j] = focal_diff*static_cast<Dtype>(label_count_data[label_value]);
	      }
        }
      }
    }
    // Scale gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_scal(prob_.count(), loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalSoftmaxLossLayer);
#endif

INSTANTIATE_CLASS(FocalSoftmaxLossLayer);
REGISTER_LAYER_CLASS(FocalSoftmaxLoss);

}  // namespace caffe
