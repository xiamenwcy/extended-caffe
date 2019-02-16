#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include <string> 

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/layers/center_layer.hpp"

namespace caffe {

template <typename Dtype>
void CenterLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";
  output_coordinate_ = this->layer_param_.center_param().output_coordinate();
  const int num_images = bottom[0]->num();
  const int num_channels = bottom[0]->channels();
  if ( output_coordinate_ ) {
	  top[0]->Reshape(num_images, num_channels, 1, 2);
  }
  else
  {
	   top[0]->ReshapeLike(*bottom[0]);
	   CHECK_EQ(bottom[0]->count(),  top[0]->count());
  }

}
template<typename Dtype>
void CenterLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top)
{
   
  has_ignore_label_ =
	  this->layer_param_.center_param().has_ignore_label();
  if ( has_ignore_label_ ) {
	  ignore_label_ = this->layer_param_.center_param().ignore_label();
  }
  
}

cv::Point find_point(const cv::Mat& src,uchar data=1)
{
	int  width = src.cols;
	int height = src.rows;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar value = src.at<uchar>(i, j);  //等价
			if (data == value)
				return cv::Point(j,i);
		}
	}
	return cv::Point(-1, -1);
}


cv::Mat gauss_heatmap(const cv::Mat& src, cv::Point my_center,float threshold =1) {

	int  width = src.cols;
	int height = src.rows;
	cv::Mat result(src.size(), CV_32FC1);  

	
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			  float distance=(i-my_center.y)*(i-my_center.y)+(j-my_center.x)*(j-my_center.x);
			  result.at<float>(i, j) = expf(-distance/ (2 * threshold*threshold));
		}
	}	
		
	return result;	
	
}



template <typename Dtype>
void CenterLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
		  
    const Dtype* gt_label = bottom[0]->cpu_data(); 
    const int num_images = bottom[0]->num();
    const int label_height = bottom[0]->height();
    const int label_width = bottom[0]->width();
    const int num_channels = bottom[0]->channels();
	const int label_channel_size = label_height * label_width;
    const int label_img_size = label_channel_size * num_channels;
	
	 
	Dtype* label_ptr = top[0]->mutable_cpu_data();
	float  threshold = this->layer_param_.center_param().gauss_sigma();  //默认为3
     
	   for (int idx_img = 0; idx_img < num_images; idx_img++)
     {
        for (int idx_ch = 0; idx_ch < num_channels; idx_ch++)
        {
			 cv::Mat label_img(label_height,label_width, CV_8UC1);
            for (int i = 0; i < label_height; i++)
            {
                for (int j = 0; j < label_width; j++)
                {
                    int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_width + j;
					const int target_value = static_cast<int>(gt_label[image_idx]);
				
				    label_img.at<uchar>(i, j)=target_value;
                }           
            }
		     cv::Point my_center = find_point(label_img,1);
			 if(output_coordinate_)
			 {
				  const int target_size = 2 * num_channels;
				  int target_index = idx_img * target_size + idx_ch * 2 ;
				  label_ptr[target_index]=my_center.x;
				  label_ptr[target_index+1]=my_center.y;
				  continue;
			 }
             //否则处理map
			 if (my_center == cv::Point(-1, -1))
			{
				
				 for (int i = 0; i < label_height; i++)
				{
						for (int j = 0; j < label_width; j++)
						{
							int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_width + j;
							label_ptr[image_idx]= gt_label[image_idx];
						}           
				}	
			 }
			 else
			 {
				 
					cv::Mat resultImg=gauss_heatmap(label_img, my_center, threshold);
					 for (int i = 0; i < label_height; i++)
						{
							for (int j = 0; j < label_width; j++)
							{
								int image_idx = idx_img * label_img_size + idx_ch * label_channel_size + i * label_width + j;
								const int target_value = static_cast<int>(gt_label[image_idx]);
							
								if (has_ignore_label_ && target_value == ignore_label_) 
								{
									label_ptr[image_idx]= gt_label[image_idx];
								}
								else
								{
									 label_ptr[image_idx]=resultImg.at<float>(i, j) ;
								}
								
								 
							}           
						}
			 }

        }
	  }  
	   
}

#ifdef CPU_ONLY
STUB_GPU(CenterLayer);
#endif


INSTANTIATE_CLASS(CenterLayer);
REGISTER_LAYER_CLASS(Center);

}  // namespace caffe
