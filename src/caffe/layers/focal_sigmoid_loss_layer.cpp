#include <vector>

#include "caffe/layers/focal_sigmoid_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void FocalSigmoidLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);
  has_ignore_label_ =
	  this->layer_param_.loss_param().has_ignore_label();
  if ( has_ignore_label_ ) {
	  ignore_label_ = this->layer_param_.loss_param().ignore_label();
  }
  bool has_alpha =this->layer_param().focal_sigmoid_loss_param().has_alpha();
  bool has_gamma =this->layer_param().focal_sigmoid_loss_param().has_gamma();
  if(has_alpha && has_gamma)
  {
	  alpha_ = this->layer_param().focal_sigmoid_loss_param().alpha();
      gamma_ = this->layer_param().focal_sigmoid_loss_param().gamma();
  }
  else
  {
	  alpha_=0.5;
	  gamma_=0.0;
	  LOG(WARNING) << "Loss params need to give alpha and gamma,or else turn to use common CrossEntropyLossLayer";
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
void FocalSigmoidLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
   inner_num_ = bottom[0]->count(2);  // h*w
   CHECK_EQ(bottom[0]->width()*bottom[0]->height(), bottom[1]->width()*bottom[1]->height()) <<
    "MULTICHANNEL_REWEIGHTED_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same spatial dimension.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
  scaler_.ReshapeLike(*bottom[0]);
  if (top.size() >= 2) {
	  // softmax output
	  top[1]->ReshapeLike(*bottom[0]);
  }

}


template <typename Dtype>
Dtype FocalSigmoidLossLayer<Dtype>::get_normalizer(
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
void FocalSigmoidLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  const Dtype* sigmoid_data = sigmoid_output_->cpu_data();
  // Compute the loss (negative log likelihood)
  const int count = bottom[0]->count();
  int dim = count/ outer_num_;
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  Dtype* loss_data = bottom[0]->mutable_cpu_diff();
  Dtype loss = 0;
  Dtype* scale = scaler_.mutable_cpu_data();
  Dtype* oriloss = scaler_.mutable_cpu_diff();
  Dtype count_pos = 0, count_neg=0;

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
  if (alpha_<0)  //表示自适应选择权重
  {	  
	  if (static_cast<int>(count_pos)== 0 || static_cast<int>(count_neg) == 0)
	  {
		  alpha_ = 0.5;
	  }
	  else
	  {
		  alpha_ = count_neg / (count_neg + count_pos);
	  }
  }
  for (int i = 0; i < outer_num_; ++i) 
	  {
		  for(int j=0;j< bottom[0]->shape(1);j++)
		  {
			  for(int k=0;k<inner_num_;k++)
			  {
				  
				   int index = i * dim + j * inner_num_ + k;
				   const int target_value = static_cast<int>(target[i * inner_num_ + k]);
				   int target_normalized_value = target_value==j+1?1:0;  //归一化到0 or1
					if (has_ignore_label_ && target_value == ignore_label_) 
					{
						scale[index] = 0;
						oriloss[index] = 0;
					}
					else
					{
		  scale[index] = (target_normalized_value == 1 ? alpha_ : 1 - alpha_) * powf(1 - (target_normalized_value == 1 ? sigmoid_data[index] : (1 - sigmoid_data[index])), gamma_);
		  oriloss[index] = -(input_data[index] * (target_normalized_value - (input_data[index] >= 0)) - log(1 + exp(input_data[index] - 2 * input_data[index] * (input_data[index] >= 0))));
		  
					}
				  } 

			  }
  
      }
  
  caffe_mul(count, scaler_.cpu_data(), scaler_.cpu_diff(), loss_data);
  loss = caffe_cpu_asum(count, loss_data);
  normalizer_ = get_normalizer(normalization_, count_pos+count_neg);
  top[0]->mutable_cpu_data()[0] = loss / normalizer_;
  if (top.size() == 2) {
	  top[1]->ShareData(*sigmoid_output_);
  }
 
}

template <typename Dtype>
void FocalSigmoidLossLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0])
  {
	   
    // First, compute the diff
    const int count = bottom[0]->count();
	int dim = count/ outer_num_;
    const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
    const Dtype* target = bottom[1]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const Dtype* scale = scaler_.cpu_data();
	 Dtype  pt=0;
	// First item: d(oriloss)*scale
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
					    bottom_diff[index] = 0; 
					}
					else
					{
						 bottom_diff[index] =  scale[index]*(pt-target_normalized_value);
					}
			   } 

		    }
       }
	
	// Second item: oriloss*d(scale)
	// save result in scaler_.data
    Dtype* secondItem = scaler_.mutable_cpu_data();
	
	for (int i = 0; i < outer_num_; ++i) 
	  {
		  for(int j=0;j< bottom[0]->shape(1);j++)
		  {
			  for(int k=0;k<inner_num_;k++)
			  {
				  int index= i * dim + j * inner_num_ + k;
				   
				   const int target_value = static_cast<int>(target[i * inner_num_ + k]);
				    int target_normalized_value = target_value==j+1?1:0;  //归一化到0 or1
					pt = (target_normalized_value == 1 ? sigmoid_output_data[index] : (1 - sigmoid_output_data[index]));
					if (has_ignore_label_ && target_value == ignore_label_) {
					    secondItem[index] = 0; 
					}
					else
					{
						secondItem[index] = (target_normalized_value == 1 ? alpha_ : 1 - alpha_) * gamma_ *
			powf(1 - pt, gamma_) * pt *(target_normalized_value == 1 ? -1 : 1);
					}
			   } 

		    }
       }
	
	caffe_mul(count, scaler_.cpu_data(), scaler_.cpu_diff(), scaler_.mutable_cpu_data());
	caffe_add(count, scaler_.cpu_data(), bottom[0]->cpu_diff(), bottom_diff);

    // Scale down gradient
	// Scale gradient
	Dtype loss_weight = top[0]->cpu_diff()[0] / normalizer_;
	caffe_scal(count, loss_weight, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(FocalSigmoidLossLayer);
#endif

INSTANTIATE_CLASS(FocalSigmoidLossLayer);
REGISTER_LAYER_CLASS(FocalSigmoidLoss);

}  // namespace caffe
