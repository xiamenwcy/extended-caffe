#include <cmath>
#include <vector>

#include "caffe/layers/slice_label_layer.hpp"

namespace caffe {

template <typename Dtype>
void SliceLabelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  count_ = bottom[0]->count();
  for (int i = 0; i < top.size(); ++i) {
    // Do not allow in-place computation in the SplitLayer. 
    CHECK_NE(top[i], bottom[0]) << this->type() << " Layer does not "
        "allow in-place computation.";
    top[i]->ReshapeLike(*bottom[0]);
    CHECK_EQ(count_, top[i]->count());
  }
}

template<typename Dtype>
void SliceLabelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
   
  SliceLabelParameter my_slice_param = this->layer_param_.slice_label_param(); 
  has_ignore_label_ = my_slice_param.has_ignore_label();
  if ( has_ignore_label_ ) {
	  ignore_label_ = my_slice_param.ignore_label();
  }

  reserve_ignore_label_.clear();
  std::copy(my_slice_param.reserve_ignore_label().begin(),
      my_slice_param.reserve_ignore_label().end(),
      std::back_inserter(reserve_ignore_label_));
	
  CHECK_EQ(reserve_ignore_label_.size(), top.size()) << "reserve_ignore_label_ must have the same dimension with top size.";
	  
}


template <typename Dtype>
void SliceLabelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  
  for (int i = 0; i < top.size(); ++i) {
     Dtype* top_data = top[i]->mutable_cpu_data();
	 caffe_set(count_,  Dtype(0), top_data);  //label_初始化为0
  }
  
  for (int i = 0; i < count_; ++i) {
	   const int label_value = static_cast<int>(bottom_data[i]);
	   if(label_value>0&&label_value<=top.size())
	   {
		   Dtype* top_data = top[label_value-1]->mutable_cpu_data();
		   top_data[i] = 1;
		   continue;
	   }
  
	   if (has_ignore_label_ && label_value == ignore_label_) 
		 {
				for (int j = 0; j < top.size(); ++j) 
				{
					bool temp_reserve_ignore_label= reserve_ignore_label_[j]!=0?true:false;
					Dtype* top_data = top[j]->mutable_cpu_data();
					if(temp_reserve_ignore_label)
					{
						top_data[i] =ignore_label_;
					}
					else
					{
						top_data[i] = 0;
					}
			    }
		 }
			   
	  }
 }


#ifdef CPU_ONLY
STUB_GPU(SliceLabelLayer);
#endif

INSTANTIATE_CLASS(SliceLabelLayer);
REGISTER_LAYER_CLASS(SliceLabel);

}  // namespace caffe
