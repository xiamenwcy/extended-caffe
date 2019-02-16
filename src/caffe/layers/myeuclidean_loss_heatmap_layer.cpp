#include <vector>

#include "caffe/layers/myeuclidean_loss_heatmap_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MyEuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
   outer_num_ = bottom[0]->shape(0);  // batch size
   inner_num_ = bottom[0]->count(1);  // h*w
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  diff_.ReshapeLike(*bottom[0]);
}

template<typename Dtype>
void MyEuclideanLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
   
  has_ignore_label_ =
	  this->layer_param_.loss_param().has_ignore_label();
  if ( has_ignore_label_ ) {
	  ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if ( this->layer_param_.loss_param().has_normalization() ) {
	  normalization_ = this->layer_param_.loss_param().normalization();
  }
  else if ( this->layer_param_.loss_param().has_normalize() ) {
	  normalization_ = this->layer_param_.loss_param().normalize() ?
	  LossParameter_NormalizationMode_VALID : LossParameter_NormalizationMode_BATCH_SIZE;
  }
  else {
	  normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}


template <typename Dtype>
Dtype MyEuclideanLossLayer<Dtype>::get_normalizer(
	LossParameter_NormalizationMode normalization_mode, int valid_count) {
	Dtype normalizer;
	switch ( normalization_mode ) {
		case LossParameter_NormalizationMode_FULL:
			normalizer = Dtype(outer_num_ * inner_num_);
			break;
		case LossParameter_NormalizationMode_VALID:
			if ( valid_count == -1 ) {
				normalizer = Dtype(outer_num_ * inner_num_);
			}
			else {
				normalizer = Dtype(valid_count);
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
void MyEuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
    Dtype loss = 0;

    const Dtype* bottom_pred = bottom[0]->cpu_data(); // predictions for all images
    const Dtype* gt_pred = bottom[1]->cpu_data();    // GT predictions

	const int count = bottom[0]->count();
	Dtype valid_count = 0;
	
	// Loop over images
    for (int i = 0; i < count; i++)
    {
		const int target_value = static_cast<int>(gt_pred[i]);
		float diff ;
		if (has_ignore_label_ && target_value == ignore_label_) 
		{
			  diff=0;
		}
		else
		{
			  diff = (float)bottom_pred[i] - (float)gt_pred[i];
			  valid_count++;
	   
		}
		 loss += diff * diff;
 
    }

    DLOG(INFO) << "total loss: " << loss;
	normalizer_ = get_normalizer(normalization_, valid_count)*2;
    loss /= normalizer_;
    DLOG(INFO) << "total normalised loss: " << loss;

    top[0]->mutable_cpu_data()[0] = loss;
	
}

template <typename Dtype>
void MyEuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		
   const int count = bottom[0]->count();
    //const int channels = bottom[0]->channels();

    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(), diff_.mutable_cpu_data());

    const Dtype alpha = top[0]->cpu_diff()[0] / normalizer_*2;
    caffe_cpu_axpby(
          count,              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[0]->mutable_cpu_diff());  // b

	
	// Zero out gradient of ignored targets.
	const Dtype* target = bottom[1]->cpu_data();	
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	for (int i = 0; i < count; ++i) {
		const int target_value = static_cast<int>(target[i]);
			if (has_ignore_label_ && target_value == ignore_label_) {
					bottom_diff[i] = 0;
			}
	}	  
}

#ifdef CPU_ONLY
STUB_GPU(MyEuclideanLossLayer);
#endif

INSTANTIATE_CLASS(MyEuclideanLossLayer);
REGISTER_LAYER_CLASS(MyEuclideanLoss);

}  // namespace caffe
