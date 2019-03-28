/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <mkldnn/include/mkldnn_types.h>
#include <memory>
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/fc_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/mkldnn_helper.h"
#include "paddle/fluid/platform/mkldnn_reuse.h"
#include "paddle/fluid/platform/variant.h"

namespace paddle {
namespace operators {

using framework::DataLayout;
using framework::Tensor;
using framework::DDim;
using framework::ExecutionContext;
using platform::MKLDNNDeviceContext;
using platform::to_void_cast;
using platform::GetMKLDNNFormat;
using mkldnn::memory;
using mkldnn::inner_product_forward;
using mkldnn::primitive;
using mkldnn::stream;
using mkldnn::prop_kind;

std::string CreateKey(const paddle::framework::ExecutionContext& ctx,
                      const std::vector<int>& src_tz,
                      const std::vector<int>& weights_tz) {
  std::string key;
  key.reserve(platform::MKLDNNHandler::MaxKeyLength);
  platform::MKLDNNHandler::AppendKeyDims(&key, src_tz);
  platform::MKLDNNHandler::AppendKeyDims(&key, weights_tz);
  platform::MKLDNNHandler::AppendKey(&key, ctx.op().Input("W"));
  return key;
}

template <typename T, typename K>
class FCPrimitiveFactory {
 public:
  explicit FCPrimitiveFactory(const mkldnn::engine& engine) : engine_(engine) {}

  inner_product_forward CreateFcPrimitive(const Tensor* input,
                                          const Tensor* weights,
                                          const Tensor* bias, Tensor* output,
                                          const ExecutionContext& ctx,
                                          bool fuse_relu) {
    std::string key =
        CreateKey(ctx, paddle::framework::vectorize2int(input->dims()),
                  paddle::framework::vectorize2int(weights->dims()));
    std::shared_ptr<mkldnn::inner_product_forward> fc_p = nullptr;
    bool is_int8 =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    if (!is_int8) {
      const std::string key_fc_primitive = key + "@fc_p";
      const std::string key_fc_src_mem_primitive = key + "@fc_src_mem_p";
      const std::string key_fc_dst_mem_primitive = key + "@fc_dst_mem_p";
      const std::string key_fc_bias_mem_primitive = key + "@fc_bias_mem_p";
      const std::string key_fc_weights_mem_primitive =
          key + "@fc_weights_mem_p";

      auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
      auto fc_p = std::static_pointer_cast<mkldnn::inner_product_forward>(
          dev_ctx.GetBlob(key_fc_primitive));
      if (fc_p) {
        auto fc_src_memory_p = std::static_pointer_cast<memory>(
            dev_ctx.GetBlob(key_fc_src_mem_primitive));
        auto fc_dst_memory_p = std::static_pointer_cast<memory>(
            dev_ctx.GetBlob(key_fc_dst_mem_primitive));

        auto fc_bias_memory_p = std::static_pointer_cast<memory>(
            dev_ctx.GetBlob(key_fc_bias_mem_primitive));
        auto fc_weights_memory_p = std::static_pointer_cast<memory>(
            dev_ctx.GetBlob(key_fc_weights_mem_primitive));

        fc_src_memory_p->set_data_handle(to_void_cast<T>(input->data<T>()));
        fc_bias_memory_p->set_data_handle(to_void_cast<T>(bias->data<T>()));
        fc_weights_memory_p->set_data_handle(
            to_void_cast<T>(weights->data<T>()));
        fc_dst_memory_p->set_data_handle(
            output->mutable_data<T>(ctx.GetPlace()));

        output->set_format((memory::format)fc_dst_memory_p->get_primitive_desc()
                               .desc()
                               .data.format);

        return *fc_p;

      } else {
        auto src_desc = CreateMemDescriptor(
            input, platform::MKLDNNGetDataType<float>(), input->format());

        auto src_memory = CreateMemory_ptr(src_desc, input);

        dev_ctx.SetBlob(key_fc_src_mem_primitive, src_memory);

        auto weights_dims = GetCorrectedWeightsDims(weights);
        auto weights_desc = CreateMemDescriptor(
            weights_dims, memory::data_type::f32, memory::format::oi);
        auto weights_memory = CreateMemory_ptr(weights_desc, weights);
        if (src_desc.data.ndims == 4) {
          weights_memory = CreateFourDimWeightsMemory(
              input, weights, ctx, key_fc_weights_mem_primitive);
          weights_desc = weights_memory->get_primitive_desc().desc();
        } else {
          dev_ctx.SetBlob(key_fc_weights_mem_primitive, weights_memory);
        }

        auto dst_desc = CreateMemDescriptor(output, memory::data_type::f32,
                                            memory::format::any);
        if (src_desc.data.ndims == 4) {
          fc_p = CreateFcPrimitive(src_desc, *src_memory, weights_desc,
                                   *weights_memory, dst_desc, bias, output, ctx,
                                   key_fc_bias_mem_primitive,
                                   key_fc_dst_mem_primitive, fuse_relu);
        } else {
          fc_p = CreateFcPrimitive(src_desc, *src_memory, weights_desc,
                                   *weights_memory, dst_desc, bias, output, ctx,
                                   key_fc_bias_mem_primitive,
                                   key_fc_dst_mem_primitive, fuse_relu);
        }

        dev_ctx.SetBlob(key_fc_primitive, fc_p);
        return *fc_p;
      }
    } else {
      auto scale_in_data = ctx.Attr<float>("Scale_in");
      auto scale_weights_data = ctx.Attr<std::vector<float>>("Scale_weights");
      bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
      bool fuse_relu = ctx.Attr<bool>("fuse_relu");
      auto scale_out_data =
          force_fp32_output ? 1.0f : ctx.Attr<float>("Scale_out");
      std::shared_ptr<mkldnn::inner_product_forward::desc> fc_desc;
      // std::shared_ptr<mkldnn::inner_product_forward::primitive_desc>
      // fc_prim_desc;
      std::string data_format = ctx.Attr<std::string>("data_format");
      auto dst_dt = fuse_relu ? paddle::framework::ToMKLDNNDataType(
                                    framework::DataTypeTrait<uint8_t>::DataType)
                              : paddle::framework::ToMKLDNNDataType(
                                    framework::DataTypeTrait<int8_t>::DataType);
      if (force_fp32_output) {
        dst_dt = paddle::framework::ToMKLDNNDataType(
            framework::DataTypeTrait<float>::DataType);
      }
      auto chosen_memory_format =
          platform::data_format_to_memory_format(data_format);
      auto src_dt = platform::MKLDNNGetDataType<T>();
      auto src_desc = CreateMemDescriptor(input, src_dt, chosen_memory_format);
      auto dst_desc = CreateMemDescriptor(output, dst_dt, chosen_memory_format);
      auto weights_dims = GetCorrectedWeightsDims(weights);
      auto weights_desc = CreateMemDescriptor(
          weights_dims, memory::data_type::s8, chosen_memory_format);
      auto user_weights_desc = CreateMemDescriptor(
          weights_dims, platform::MKLDNNGetDataType<float>(),
          memory::format::oi);
      if (src_desc.data.ndims == 4) {
        auto input_dims = framework::vectorize2int(input->dims());
        auto weight_dims = framework::vectorize2int(weights->dims());
        auto dims = {weight_dims[1], input_dims[1], input_dims[2],
                     input_dims[3]};
        weights_desc = CreateMemDescriptor(dims, memory::data_type::s8,
                                           chosen_memory_format);
        user_weights_desc = CreateMemDescriptor(
            dims, platform::MKLDNNGetDataType<float>(), memory::format::oihw);
      }

      std::vector<int> weights_tz =
          paddle::framework::vectorize2int(weights->dims());
      bool is_multi_channel = scale_weights_data.size() > 1;
      int weights_mask_reorder = is_multi_channel ? 1 << 0 : 0;
      int count = is_multi_channel ? scale_weights_data.size() : 1;
      std::vector<float> output_shift_scale(count);
#pragma omp parallel for if (count > 1)
      for (int i = 0; i < count; i++) {
        if (scale_weights_data[i] == 0.0)
          output_shift_scale[i] = scale_out_data;
        else
          output_shift_scale[i] =
              scale_out_data / (scale_in_data * scale_weights_data[i]);
      }

      if (bias) {
        auto bias_desc = CreateMemDescriptor(bias, memory::data_type::s32,
                                             memory::format::x);
        fc_desc.reset(new inner_product_forward::desc(
            prop_kind::forward, src_desc, weights_desc, bias_desc, dst_desc));
      } else {
        fc_desc.reset(new inner_product_forward::desc(
            prop_kind::forward, src_desc, weights_desc, dst_desc));
      }

      mkldnn::primitive_attr fc_attr;
      mkldnn::post_ops post_operations;
      int out_mask = output_shift_scale.size() > 1 ? 1 << 1 : 0;
      fc_attr.set_output_scales(out_mask, output_shift_scale);
      if (fuse_relu) {
        constexpr float scale = 1.0f;
        constexpr float negative_slope = 0.0f;
        constexpr float placeholder = 1.0f;  // beta
        post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                       negative_slope, placeholder);
      }
      fc_attr.set_post_ops(post_operations);

      auto fc_prim_desc = mkldnn::inner_product_forward::primitive_desc(
          *fc_desc, fc_attr, engine_);

      auto user_src_desc = CreateMemDescriptor(
          input, platform::MKLDNNGetDataType<T>(), input->format());
      auto user_src_pd = memory::primitive_desc(user_src_desc, engine_);
      auto src_pd = fc_prim_desc.src_primitive_desc();
      input_ = Reorder(user_src_pd, src_pd, input->data<T>());

      auto weights_pd = fc_prim_desc.weights_primitive_desc();
      auto user_weights_pd = memory::primitive_desc(user_weights_desc, engine_);
      weights_ = Reorder(user_weights_pd, weights_pd, weights->data<K>(),
                         is_int8, weights_mask_reorder, scale_weights_data);
      weights_desc = weights_.get().get_primitive_desc().desc();
      if (bias) {
        int bias_mask_reorder = is_multi_channel ? 1 << 0 : 1;
        std::vector<float> bias_scale(count);
#pragma omp parallel for if (count > 1)
        for (int i = 0; i < count; i++) {
          if (scale_weights_data[i] == 0) {
            bias_scale[i] = 1.0f;
          } else {
            bias_scale[i] = scale_in_data * scale_weights_data[i];
          }
        }

        const K* bias_data = bias->data<K>();
        auto user_bias_desc = CreateMemDescriptor(bias, memory::data_type::f32,
                                                  memory::format::x);
        auto user_bias_pd = memory::primitive_desc(user_bias_desc, engine_);
        auto bias_pd = fc_prim_desc.bias_primitive_desc();
        bias_ = Reorder(user_bias_pd, bias_pd, bias_data, is_int8,
                        bias_mask_reorder, bias_scale);
        output_ = CreateDstMemory(fc_prim_desc, ctx, output);

        fc_p = std::make_shared<inner_product_forward>(
            fc_prim_desc, *input_, *weights_, *bias_, *output_);
      } else {
        output_ = CreateDstMemory(fc_prim_desc, ctx, output);

        fc_p = std::make_shared<inner_product_forward>(fc_prim_desc, *input_,
                                                       *weights_, *output_);
      }
    }
    return *fc_p;
  }

 private:
  bool IsOutputSame(Tensor* out, const ExecutionContext& ctx) {
    return output_->get_data_handle() == out->mutable_data<T>(ctx.GetPlace());
  }

  memory::format MatchWeightFormat(memory::format fmt) {
    using format = memory::format;
    switch (fmt) {
      case format::nChw16c:
        return format::oIhw16i;
      case format::nChw8c:
        return format::oIhw8i;
      case format::nchw:
        return format::oihw;
      default:
        return format::format_undef;
    }
  }

  mkldnn::memory Reorder(const memory::desc& src_desc,
                         const memory::desc& dst_desc, const void* src_data) {
    auto src_mem = memory({src_desc, engine_}, const_cast<void*>(src_data));
    auto dst_mem = memory({dst_desc, engine_});
    auto reorder = mkldnn::reorder(src_mem, dst_mem);
    stream(stream::kind::eager).submit({reorder}).wait();
    return dst_mem;
  }

  mkldnn::memory Reorder(const memory::primitive_desc& src_desc,
                         const memory::primitive_desc& dst_desc,
                         const void* src_data, bool is_int8 = false,
                         int mask = 0,
                         const std::vector<float> scale_data = {1.0f}) {
    auto src_mem = memory(src_desc, const_cast<void*>(src_data));
    auto dst_mem = memory(dst_desc);

    if (is_int8) {
      mkldnn::primitive_attr attri;
      attri.set_output_scales(mask, scale_data);
      auto reorder_pd = std::shared_ptr<mkldnn::reorder::primitive_desc>(
          new mkldnn::reorder::primitive_desc(src_desc, dst_desc, attri));
      auto reorder = mkldnn::reorder(*reorder_pd, src_mem, dst_mem);
      stream(stream::kind::eager).submit({reorder}).wait();
    } else {
      auto reorder = mkldnn::reorder(src_mem, dst_mem);
      stream(stream::kind::eager).submit({reorder}).wait();
    }
    return dst_mem;
  }
  static mkldnn::memory::desc CreateMemDescriptor(const std::vector<int>& dims,
                                                  mkldnn::memory::data_type dt,
                                                  memory::format format) {
    return platform::MKLDNNMemDesc(dims, dt, format);
  }

  static mkldnn::memory::desc CreateMemDescriptor(const Tensor* tensor,
                                                  mkldnn::memory::data_type dt,
                                                  memory::format format) {
    auto dims = framework::vectorize2int(tensor->dims());
    return CreateMemDescriptor(dims, dt, format);
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const Tensor* tensor) {
    return CreateMemory(desc, tensor->data<T>());
  }

  mkldnn::memory CreateMemory(const mkldnn::memory::desc& desc,
                              const void* data) {
    return memory({desc, engine_}, const_cast<void*>(data));
  }

  std::shared_ptr<mkldnn::memory> CreateMemory_ptr(
      const mkldnn::memory::desc& desc, const Tensor* tensor) {
    return CreateMemory_ptr(desc, tensor->data<T>());
  }

  std::shared_ptr<mkldnn::memory> CreateMemory_ptr(
      const mkldnn::memory::desc& desc, const void* data) {
    auto src_memory_desc = memory::primitive_desc(desc, engine_);
    return std::make_shared<mkldnn::memory>(src_memory_desc,
                                            const_cast<void*>(data));
  }

  std::shared_ptr<mkldnn::inner_product_forward> CreateFcPrimitive(
      const memory::desc& src_desc, const memory& src_memory,
      const memory::desc& weights_desc, const memory& weights_memory,
      const memory::desc& dst_desc, const Tensor* bias, Tensor* output,
      const ExecutionContext& ctx, const std::string& bias_key,
      const std::string& dst_key, bool fuse_relu) {
    if (bias) {
      auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();

      auto bias_desc = CreateMemDescriptor(
          bias, platform::MKLDNNGetDataType<float>(), bias->format());
      // bias_ = CreateMemory(bias_desc, bias);
      auto bias_mem = CreateMemory_ptr(bias_desc, bias);
      dev_ctx.SetBlob(bias_key, bias_mem);

      auto fc_prim_desc = CreateFcPrimDesc(src_desc, weights_desc, bias_desc,
                                           dst_desc, fuse_relu);
      // const std::string key_fc_dst_mem_primitive = key + "@fc_dst_mem_p";
      // output_ = CreateDstMemory(fc_prim_desc, ctx, output);
      auto output_mem = CreateDstMemory_ptr(fc_prim_desc, ctx, output, dst_key);
      dev_ctx.SetBlob(dst_key, output_mem);

      return std::make_shared<inner_product_forward>(
          fc_prim_desc, src_memory, weights_memory, *bias_mem, *output_mem);
    } else {
      auto fc_prim_desc =
          CreateFcPrimDesc(src_desc, weights_desc, dst_desc, fuse_relu);

      output_ = CreateDstMemory(fc_prim_desc, ctx, output);
      return std::make_shared<inner_product_forward>(fc_prim_desc, src_memory,
                                                     weights_memory, *output_);
    }
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& bias_desc,
      const mkldnn::memory::desc& dst_desc, bool fuse_relu) {
    mkldnn::primitive_attr conv_attr = CreatePostOps(fuse_relu);

    auto fc_desc = inner_product_forward::desc(
        prop_kind::forward, input_desc, weights_desc, bias_desc, dst_desc);

    return inner_product_forward::primitive_desc(fc_desc, conv_attr, engine_);
  }

  mkldnn::inner_product_forward::primitive_desc CreateFcPrimDesc(
      const mkldnn::memory::desc& input_desc,
      const mkldnn::memory::desc& weights_desc,
      const mkldnn::memory::desc& dst_desc, bool fuse_relu) {
    mkldnn::primitive_attr conv_attr = CreatePostOps(fuse_relu);
    auto fc_desc = inner_product_forward::desc(prop_kind::forward, input_desc,
                                               weights_desc, dst_desc);
    return inner_product_forward::primitive_desc(fc_desc, conv_attr, engine_);
  }

  std::vector<int> GetCorrectedWeightsDims(const Tensor* weights) {
    // MKLDNN requires weights layout to be column major.
    // The values have already been transposed, but the shape needs to be fixed.
    // It cannot be done during an earlier stage since InferShape verifies
    // dimensions assuming the weights weren't transposed.
    std::vector<int> weights_dims =
        paddle::framework::vectorize2int(weights->dims());

    std::swap(weights_dims[0], weights_dims[1]);
    return weights_dims;
  }

  std::shared_ptr<mkldnn::memory> CreateFourDimWeightsMemory(
      const Tensor* input, const Tensor* weights, const ExecutionContext& ctx,
      const std::string& key) {
    auto input_dims = framework::vectorize2int(input->dims());
    auto weight_dims = framework::vectorize2int(weights->dims());
    auto dims = {weight_dims[1], input_dims[1], input_dims[2], input_dims[3]};

    auto dst_format = MatchWeightFormat(input->format());
    auto src_desc = CreateMemDescriptor(dims, platform::MKLDNNGetDataType<K>(),
                                        memory::format::oihw);
    auto dst_desc =
        CreateMemDescriptor(dims, platform::MKLDNNGetDataType<K>(), dst_format);
#if 1
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    auto data = std::static_pointer_cast<mkldnn::memory>(dev_ctx.GetBlob(key));

    auto dst_mem = Reorder(src_desc, dst_desc, weights->data<T>());

    data = std::make_shared<mkldnn::memory>(
        dst_mem.get_primitive_desc(), to_void_cast<T>(weights->data<T>()));
    dev_ctx.SetBlob(key, data);
    return data;

#else
    return Reorder(src_desc, dst_desc, weights->data<T>());
#endif
  }

  mkldnn::memory CreateDstMemory(
      const mkldnn::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx, Tensor* output) {
    auto dst_prim_desc = fc_prim_desc.dst_primitive_desc();
    auto buffer_size = dst_prim_desc.get_size();
    bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    bool is_int8 =
        std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value;
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    if (is_int8) {
      if (force_fp32_output) {
        float* output_data = output->mutable_data<float>(
            ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);

        output->set_format((memory::format)dst_prim_desc.desc().data.format);
        return memory(dst_prim_desc, to_void_cast<float>(output_data));
      } else if (fuse_relu) {
        uint8_t* output_data = output->mutable_data<uint8_t>(
            ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);

        output->set_format((memory::format)dst_prim_desc.desc().data.format);
        return memory(dst_prim_desc, to_void_cast<uint8_t>(output_data));
      } else {
        int8_t* output_data = output->mutable_data<int8_t>(
            ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);

        output->set_format((memory::format)dst_prim_desc.desc().data.format);
        return memory(dst_prim_desc, to_void_cast<int8_t>(output_data));
      }
    }
    T* output_data = output->mutable_data<T>(
        ctx.GetPlace(), ::paddle::memory::Allocator::kDefault, buffer_size);
    output->set_format((memory::format)dst_prim_desc.desc().data.format);
    return memory(dst_prim_desc, to_void_cast<T>(output_data));
  }

  std::shared_ptr<mkldnn::memory> CreateDstMemory_ptr(
      const mkldnn::inner_product_forward::primitive_desc& fc_prim_desc,
      const ExecutionContext& ctx, Tensor* output, const std::string& key) {
    auto dst_prim_desc = fc_prim_desc.dst_primitive_desc();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    output->set_format((memory::format)dst_prim_desc.desc().data.format);
    return std::make_shared<mkldnn::memory>(dst_prim_desc,
                                            to_void_cast<T>(output_data));
  }

  mkldnn::primitive_attr CreatePostOps(bool fuse_relu) const {
    mkldnn::primitive_attr conv_attr;
    mkldnn::post_ops post_operations;

    if (fuse_relu) {
      constexpr float scale = 1.0f;
      constexpr float negative_slope = 0.0f;
      constexpr float placeholder = 0.0f;
      post_operations.append_eltwise(scale, mkldnn::algorithm::eltwise_relu,
                                     negative_slope, placeholder);
    }
    conv_attr.set_post_ops(post_operations);
    return conv_attr;
  }

 private:
  const mkldnn::engine& engine_;
  boost::optional<memory> bias_;
  boost::optional<memory> input_;
  boost::optional<memory> output_;
  boost::optional<memory> weights_;
  boost::optional<inner_product_forward> fc_;
};

static std::string GetHash(const Tensor* weights, const bool is_s8,
                           const bool is_u8, const std::string& suffix) {
  auto dim2str = [](const DDim& operand_dims) {
    std::string str = "";
    for (size_t i = 0; i < operand_dims.size(); ++i) {
      str += std::to_string(operand_dims[i]) + "-";
    }
    return str;
  };
  std::string type_str = "";
  if (is_s8) {
    type_str = "S8";
  } else if (is_u8) {
    type_str = "U8";
  } else {
    type_str = "FP32";
  }
  return type_str + dim2str(weights->dims()) + suffix;
}

template <typename T, typename K>
std::shared_ptr<FCPrimitiveFactory<T, K>> GetPrimitiveFactory(
    const MKLDNNDeviceContext& dev_ctx, const ExecutionContext& ctx,
    const Tensor* weights, const mkldnn::engine& mkldnn_engine) {
  bool is_u8 = std::is_same<T, uint8_t>::value;
  bool is_s8 = std::is_same<T, int8_t>::value;
  const std::string key =
      GetHash(weights, is_s8, is_u8, ctx.op().Output("Out"));

  auto prim_creator =
      std::static_pointer_cast<FCPrimitiveFactory<T, K>>(dev_ctx.GetBlob(key));
  if (prim_creator == nullptr) {
    prim_creator = std::make_shared<FCPrimitiveFactory<T, K>>(mkldnn_engine);
    dev_ctx.SetBlob(key, prim_creator);
  }

  return prim_creator;
}

template <typename T, typename K>
class FCMKLDNNOpKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_cpu_place(ctx.GetPlace()),
                   "It must use CPUPlace.");
    auto& dev_ctx = ctx.template device_context<MKLDNNDeviceContext>();
    const auto& mkldnn_engine = dev_ctx.GetEngine();
    auto input = ctx.Input<framework::LoDTensor>("Input");
    /* std::cout <<"---------------input data-----------"<<std::endl;
     bool is_int8 = std::is_same<T, int8_t>::value || std::is_same<T,
     uint8_t>::value;
     if(std::is_same<T, int8_t>::value){
       const int8_t * input_data = input->data<int8_t>();
       for(int i = 0; i < input->numel(); i++){
        printf("%d  ",input_data[i]);


       }
     }
     else if(std::is_same<T, uint8_t>::value){
       const uint8_t * input_data = input->data<uint8_t>();
       for(int i = 0; i < input->numel(); i++){
         printf("%d  ",input_data[i]);
       }

     }else{
       const float * input_data = input->data<float>();
       for(int i = 0; i < input->numel(); i++){
         printf("%f  ",input_data[i]);
       }
     }
     std::cout << std::endl;*/

    auto w = ctx.Input<framework::LoDTensor>("W");
    auto bias = ctx.Input<framework::LoDTensor>("Bias");
    auto output = ctx.Output<framework::LoDTensor>("Out");
    int in_num_col_dims = ctx.Attr<int>("in_num_col_dims");
    std::vector<int64_t> output_dims;
    FCOutputSize(input->dims(), w->dims(), output_dims, in_num_col_dims);
    output->Resize(framework::make_ddim(output_dims));
    output->set_lod(input->lod());

    bool fuse_relu = ctx.Attr<bool>("fuse_relu");

    auto prim_creator =
        GetPrimitiveFactory<T, K>(dev_ctx, ctx, w, mkldnn_engine);
    auto fc =
        prim_creator->CreateFcPrimitive(input, w, bias, output, ctx, fuse_relu);
    stream(stream::kind::eager).submit({fc}).wait();
    output->set_layout(DataLayout::kMKLDNN);
    /*bool force_fp32_output = ctx.Attr<bool>("force_fp32_output");
    bool fuse_relu = ctx.Attr<bool>("fuse_relu");
    std::cout<<"-------------output data----------"<<std::endl;
    if(force_fp32_output || !is_int8){
      const float* output_data = output->data<float>();
      for(int i = 0; i < output->numel(); i++){
       printf("%f  ",output_data[i]);
      }
      std::cout << std::endl;
    }else if(fuse_relu){
      const uint8_t* output_data = output->data<uint8_t>();
      for(int i = 0; i < output->numel(); i++){
       printf("%d  ",output_data[i]);
      }
      std::cout << std::endl;
   }else{
      const int8_t* output_data = output->data<int8_t>();
      for(int i = 0; i < output->numel(); i++){
               printf("%d  ",output_data[i]);
      }
      std::cout << std::endl;
   }*/
  }

 private:
  mkldnn::inner_product_backward_weights::primitive_desc
  FcBwdWeightsPrimitiveDesc(
      const mkldnn::memory::desc& src, const mkldnn::memory::desc& diff_weights,
      const mkldnn::memory::desc& diff_dst, const mkldnn::memory::desc& bias,
      const bool with_bias,
      const mkldnn::inner_product_forward::primitive_desc& pd,
      const mkldnn::engine& engine) const {
    auto bwd_weight_desc = with_bias
                               ? mkldnn::inner_product_backward_weights::desc(
                                     src, diff_weights, bias, diff_dst)
                               : mkldnn::inner_product_backward_weights::desc(
                                     src, diff_weights, diff_dst);

    return mkldnn::inner_product_backward_weights::primitive_desc(
        bwd_weight_desc, engine, pd);
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    FP32, ops::kFCMKLDNNFP32,
                                    ops::FCMKLDNNOpKernel<float, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    U8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<uint8_t, float>);

REGISTER_OP_KERNEL_WITH_CUSTOM_TYPE(fc, MKLDNN, ::paddle::platform::CPUPlace,
                                    S8, ops::kFCMKLDNNINT8,
                                    ops::FCMKLDNNOpKernel<int8_t, float>);
