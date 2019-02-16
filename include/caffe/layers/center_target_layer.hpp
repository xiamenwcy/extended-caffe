#ifndef CAFFE_CENTER_TARGET_LAYER_HPP_
#define CAFFE_CENTER_TARGET_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Reshapes the input Blob into flat vectors.
 *
 * Note: 主要用来输出gauss化的center点map或者输出坐标点
 */
template <typename Dtype>
class CenterTargetLayer : public Layer<Dtype> {
 public:
  explicit CenterTargetLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CenterTarget"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int ExactNumTopBlobs() const { return 4; }

 protected:

  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){}
	
 
};

}  // namespace caffe

#endif  // CAFFE_CENTER_TARGET_LAYER_HPP_
