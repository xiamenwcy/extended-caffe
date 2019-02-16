#ifndef CAFFE_SLICELABEL_LAYER_HPP_
#define CAFFE_SLICELABEL_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Sigmoid function non-linearity @f$
 *         y = (1 + \exp(-x))^{-1}
 *     @f$, a classic choice in neural networks.
 *
 * Note that the gradient vanishes as the values move away from 0.
 * The ReLULayer is often a better choice for this reason.
 */
template <typename Dtype>
class SliceLabelLayer : public Layer<Dtype> {
 public:
  explicit SliceLabelLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
	  
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                          const vector<Blob<Dtype>*>& top);	  	  

  virtual inline const char* type() const { return "SliceLabel"; }
  virtual inline int ExactNumBottomBlobs() const { return 1; }
  virtual inline int MinTopBlobs() const { return 1; }
  virtual inline int MaxTopBlobs() const { return 3; }

 protected:
   virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
   virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {}
  	
  vector<int> reserve_ignore_label_;	
  bool has_ignore_label_;
  /// The label indicating that an instance should be ignored.
  int ignore_label_;
  
  int count_;
};

}  // namespace caffe

#endif  // CAFFE_SLICELABEL_LAYER_HPP_
