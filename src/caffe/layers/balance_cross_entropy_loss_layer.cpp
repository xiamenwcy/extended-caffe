#include <algorithm>
#include <vector>

#include "caffe/layers/balance_cross_entropy_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void BalanceCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  has_ignore_label_ =
    this->layer_param_.loss_param().has_ignore_label();
  if (has_ignore_label_) {
    ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  if (this->layer_param_.loss_param().has_normalization()) {
    normalization_ = this->layer_param_.loss_param().normalization();
  } else if (this->layer_param_.loss_param().has_normalize()) {
    normalization_ = this->layer_param_.loss_param().normalize() ?
                     LossParameter_NormalizationMode_VALID :
                     LossParameter_NormalizationMode_BATCH_SIZE;
  } else {
    normalization_ = LossParameter_NormalizationMode_BATCH_SIZE;
  }
}

template <typename Dtype>
void BalanceCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(2);  // h*w
  CHECK_EQ(bottom[0]->width()*bottom[0]->height(), bottom[1]->width()*bottom[1]->height()) <<
    "MULTICHANNEL_REWEIGHTED_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same spatial dimension.";
	  
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
Dtype BalanceCrossEntropyLossLayer<Dtype>::get_normalizer(
    LossParameter_NormalizationMode normalization_mode, int valid_count) {
  Dtype normalizer;
  switch (normalization_mode) {
    case LossParameter_NormalizationMode_FULL:
      normalizer = Dtype(outer_num_ * inner_num_);
      break;
    case LossParameter_NormalizationMode_VALID:
      if (valid_count == -1) {
        normalizer = Dtype(outer_num_ * inner_num_);
      } else {
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
void BalanceCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  
  int dim = bottom[0]->count() / outer_num_;
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype loss = 0, loss_pos = 0, loss_neg = 0;
  Dtype count_pos = 0, count_neg = 0;
 
  for (int i = 0; i < outer_num_; ++i) {
        for (int j = 0; j < inner_num_; j++) {
			 const int label_value = static_cast<int>(target[i * inner_num_ + j]);
			  if (has_ignore_label_ && label_value == ignore_label_) {
				  continue;
			  }
			  if (label_value!=0) {
				  count_pos++;
			  }
			  else 
			  {
				  count_neg++;
			  }
		  }
	  }
 
  for (int i = 0; i < outer_num_; ++i) 
	  {
		  for(int j=0;j< bottom[0]->shape(1);j++)
		  {
			  for(int k=0;k<inner_num_;k++)
			  {
				  
				   int index= i * dim + j * inner_num_ + k;
				   const int target_value = static_cast<int>(target[i * inner_num_ + k]);
					if (has_ignore_label_ && target_value == ignore_label_) {
					  continue;
					}
					if (target_value == j+1) {
					  loss_pos -= input_data[index] * (1 - (input_data[index] >= 0)) -
						  log(1 + exp(input_data[index] - 2 * input_data[index] * (input_data[index] >= 0)));
					} else {
					  loss_neg -= input_data[index] * (0 - (input_data[index] >= 0)) -
						  log(1 + exp(input_data[index] - 2 * input_data[index] * (input_data[index] >= 0)));
					} 
				  } 

			  }
  
      }

  if(static_cast<int>(count_pos)==0||static_cast<int>(count_neg)==0)
  {
	 alpha_=0.5;
  }
  else
  {
	  alpha_=count_neg/(count_neg+count_pos);
  }
  
  loss += loss_pos * alpha_;
  loss += loss_neg * (1-alpha_);
  normalizer_ = get_normalizer(normalization_, count_pos+count_neg);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
}

template <typename Dtype>
void BalanceCrossEntropyLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    // First, compute the diff
    const int count = bottom[0]->count();
	int dim = count/ outer_num_;
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	Dtype pt=0;
  
    for (int i = 0; i < outer_num_; ++i) 
	  {
		  for(int j=0;j< bottom[0]->shape(1);j++)
		  {
			  for(int k=0;k<inner_num_;k++)
			  {
				  int index= i * dim + j * inner_num_ + k;
				   pt = sigmoid_output_data[index];
				   const int target_value = static_cast<int>(target[i * inner_num_ + k]);
				   int target_normalized_value = target_value==j+1?1:0;  //归一化到0 or1
					if (has_ignore_label_ && target_value == ignore_label_) {
					    bottom_diff[index] = 0; continue;
					}
					if (target_normalized_value == 1) {
					   bottom_diff[index] =  alpha_*(pt-target_normalized_value);
					} else {
					   bottom_diff[index] =  (1-alpha_)*(pt-target_normalized_value);
					} 
			  } 

		   }
  
     }
	
    // Scale down gradient
    Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
    caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(BalanceCrossEntropyLossLayer);
#endif

INSTANTIATE_CLASS(BalanceCrossEntropyLossLayer);
REGISTER_LAYER_CLASS(BalanceCrossEntropyLoss);

}  // namespace caffe
