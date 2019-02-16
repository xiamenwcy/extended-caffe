#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <string> 


#include "caffe/layers/center_target_layer.hpp"

namespace caffe {

template <typename Dtype>
void CenterTargetLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		  
  const int num_images = bottom[0]->num();
  const int num_channels = bottom[0]->channels();
  CHECK_EQ(num_channels, 1) <<
    "Bottom channel must be 1.";
  const int inner_num = bottom[0]->count(1);
  CHECK_EQ(inner_num, 2) <<
    "Bottom inner_num(channel*height*width) must be 2.";
  vector<int> top_shape(2);
  top_shape[0]=num_images;
  top_shape[1]=1;
  top[0]->Reshape(top_shape);
  top_shape[1]=2;
  top[1]->Reshape(top_shape);
  top[2]->Reshape(top_shape);
  top[3]->Reshape(top_shape);
  CHECK_EQ(top[1]->count(), bottom[0]->count());
  CHECK_EQ(top[2]->count(), bottom[0]->count());
  CHECK_EQ(top[3]->count(), bottom[0]->count());
}


template <typename Dtype>
void CenterTargetLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		  
    const Dtype* my_center = bottom[0]->cpu_data(); 
    const int num_images = bottom[0]->num();
	const int inner_num = bottom[0]->count(1);  //=2
	
	 
	Dtype* label_ptr = top[0]->mutable_cpu_data();
	Dtype* target_ptr = top[1]->mutable_cpu_data();
	Dtype* inner_weight_ptr = top[2]->mutable_cpu_data();
	Dtype* outer_weight_ptr = top[3]->mutable_cpu_data();
	
     
	   for (int idx_img = 0; idx_img < num_images; idx_img++)
		 {
				 int x_idx = idx_img * inner_num +0;
				 int y_idx = idx_img * inner_num +1;
				 const int x =  static_cast<int>(my_center[x_idx]);
				 const int y =  static_cast<int>(my_center[y_idx]);
				 
				 if(x<0&& y<0) //其实二者都为-1
				 {
					 label_ptr[idx_img]=0;
					 target_ptr[x_idx] = my_center[x_idx];
					 target_ptr[y_idx] = my_center[y_idx];
					 inner_weight_ptr[x_idx] = 0;
					 inner_weight_ptr[y_idx] = 0;
					 /* outer_weight_ptr[x_idx] = 0;
					 outer_weight_ptr[y_idx] = 0; */
				 }
				else
				 {
					 label_ptr[idx_img]=1;
					 target_ptr[x_idx] = my_center[x_idx];
					 target_ptr[y_idx] = my_center[y_idx];
					 inner_weight_ptr[x_idx] = 1;
					 inner_weight_ptr[y_idx] = 1;
				/* 	 outer_weight_ptr[x_idx] = 1.0/num_images;
					 outer_weight_ptr[y_idx] = 1.0/num_images; */
				 }
				  	 outer_weight_ptr[x_idx] = 1.0/num_images;
					 outer_weight_ptr[y_idx] = 1.0/num_images; 
		  }  
	   
}

#ifdef CPU_ONLY
STUB_GPU(CenterTargetLayer);
#endif


INSTANTIATE_CLASS(CenterTargetLayer);
REGISTER_LAYER_CLASS(CenterTarget);

}  // namespace caffe
