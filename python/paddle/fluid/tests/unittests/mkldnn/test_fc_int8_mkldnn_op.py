# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import numpy as np
from  paddle.fluid.tests.unittests.op_test import OpTest

def fully_connected_naive_2dinput(src, weights, bias_data=[]):
    in_n, in_c = src.shape
    w_h, w_c = weights.shape

    # this transpose should be implemented at C code
    x_data = src
    w_data = np.transpose(np.reshape(weights, (w_c, in_c)))
    result = None

    if  not len(bias_data):
        result = np.dot(x_data, w_data)
    else:
        result = np.dot(x_data, w_data) + bias_data
    return result


def fully_connected_naive(src, weights, bias_data=[]):
    w_h, w_c = weights.shape
    if len(src.shape) == 2:
        print (src.shape)
	in_n, in_c = src.shape
        x_data = src
        w_data = np.transpose(np.reshape(weights, (w_c, in_c)))
    else:
    	in_n, in_c, in_h, in_w = src.shape
        x_data = np.reshape(src, [in_n, in_c * in_h * in_w])
        w_data = np.transpose(np.reshape(weights, (w_c, in_c * in_h * in_w)))
    #x_data = np.reshape(src, [in_n, in_c * in_h * in_w])
    # this transpose should be implemented at C code
    #w_data = np.transpose(np.reshape(weights, (w_c, in_c * in_h * in_w)))
    result = None

    if  not len(bias_data):
        result = np.dot(x_data, w_data)
    else:
        result = np.dot(x_data, w_data) + bias_data
    return result

class MatrixGenerate:
    def __init__(self, mb, ic, oc, h, w, srctype, is_2d = False, weighttype = np.float32):
	if srctype == np.uint8:
            if is_2d:
            	self.src = np.random.randint(0, 2,
                                      [mb, ic]).astype(srctype)
            else:
            	self.src = np.random.randint(0, 2,
                                      [mb, ic, h, w]).astype(srctype)
        else:
            if is_2d:
		self.src = np.random.randint(-5, 5,
                                      [mb, ic]).astype(srctype)
            else:
            	self.src = np.random.randint(-5, 5,
                                      [mb, ic, h, w]).astype(srctype)
        if is_2d:
	    self.weights = np.random.random([ic, oc]).astype(weighttype)
        else:
            self.weights = np.random.random([ic * h * w, oc]).astype(weighttype)
        self.bias = np.random.random([oc]).astype(np.float32)

class TestFCMKLDNNOp(OpTest):
    def setUp(self):
        self.op_type = "fc"
        self.use_mkldnn = True
        self.data_format = "AnyLayout"
        self.init_fuse_relu()
        self.init_src_dims()
        self.init_force_fp32_output()
        self.init_data_type()
        self.weighttype= np.float32
        self.init_data()
        self.src = self.matrix.src
        self.weights = self.matrix.weights       
        self.bias = self.matrix.bias
        self.scale_in = 1.0
        self.scale_out = 1.0 if self.force_fp32_output else 5.1
        self.scale_weights = [10.0]
        self.init_bias()
        if self.srctype == np.int8:
	    input_shift = (np.ones(self.src.shape) * 128).astype(np.uint8)
            weights_int = np.round(self.weights * self.scale_weights[0] *
                                  0.5).astype(np.int32)
            scale_output_shift = self.scale_out / (self.scale_in *
                                                   self.scale_weights[0] * 0.5)
            if self.with_bias:
                bias_int = (self.scale_in * self.scale_weights[0] 
				* self.bias * 0.5).astype(np.int32)
 		output1 = fully_connected_naive(
                         np.round((self.src + input_shift) *
                         self.scale_in).astype(np.int32),
                         weights_int, bias_int).astype(np.float32) * scale_output_shift
                output2 = fully_connected_naive(np.round((input_shift) * self.scale_in).astype(np.int32),
                         weights_int).astype(np.float32) * scale_output_shift
            else: 
                output1 = fully_connected_naive(
                         np.round((self.src + input_shift) *
                         self.scale_in).astype(np.int32), 
			 weights_int).astype(np.float32) * scale_output_shift
              
                output2 = fully_connected_naive(
                         np.round((input_shift) * self.scale_in).astype(np.int32),
                         weights_int).astype(np.float32) * scale_output_shift
            #print("self.with_bias" + str(self.with_bias))
            #print("\noutput1="+str(output1)+" \noutput2="+str(output2)+"\ndst_type"+str(self.dsttype))
            if self.force_fp32_output:
		   output = (output1 - output2).astype(self.dsttype)
                   if self.fuse_relu:
			output = np.maximum(output, 0).astype(self.dsttype)
            elif self.fuse_relu:
                    output_tmp = np.round(output1 - output2)
                    output = np.maximum(output_tmp, 0)
                    output = np.minimum(output, 255).astype(self.dsttype)
            else:
                    output_tmp = np.round(output1 - output2).astype(self.dsttype)
                    output = np.maximum(output_tmp, -128)
                    output = np.minimum(output, 127).astype(self.dsttype)

        else:
            weights_int = np.round(self.weights *
                                  self.scale_weights[0]).astype(np.int32)
            scale_output_shift = self.scale_out / (self.scale_in *
                                                   self.scale_weights[0])
            if self.with_bias:
                bias_int = (self.scale_in * self.scale_weights[0]
                                * self.bias * 0.5).astype(np.int32)
                output1 = fully_connected_naive(
                          self.src.astype(np.int32), weights_int, bias_int).astype(np.float32)
            else:
           	output1 = fully_connected_naive(
                          self.src.astype(np.int32), weights_int).astype(np.float32)
          
            output2 = output1 * scale_output_shift
            #print("\noutput1="+str(output1)+" \noutput2="+str(output2)+"\ndst_type"+str(self.dsttype))
            if self.force_fp32_output:
                   output = output2.astype(self.dsttype)
                   if self.fuse_relu:
                        output = np.maximum(output, 0).astype(self.dsttype)
            elif self.fuse_relu:
                    output = np.round(np.maximum(output2,0))
                    output = np.minimum(output,255).astype(self.dsttype)
            else:
                    output = np.round(output2).astype(self.dsttype)
                    output = np.round(np.maximum(output2,-128))
                    output = np.minimum(output,127).astype(self.dsttype) 
        print (output)
        # output=format_reorder(output1,self.size)
        if self.with_bias:
		self.inputs = {'Input':OpTest.np_dtype_to_fluid_dtype(self.src.astype(self.srctype)),
                               'W':OpTest.np_dtype_to_fluid_dtype(self.weights.astype(self.weighttype)),
                               'Bias':OpTest.np_dtype_to_fluid_dtype(self.bias.astype(self.weighttype))}
        else:
                self.inputs = {'Input':OpTest.np_dtype_to_fluid_dtype(self.src.astype(self.srctype)), 
                               'W':OpTest.np_dtype_to_fluid_dtype(self.weights.astype(self.weighttype))}

        self.outputs = {
            'Out': output
                       }

        self.attrs = {'use_mkldnn': self.use_mkldnn,
                      'Scale_in': self.scale_in,
                      'Scale_out': self.scale_out,
                      'Scale_weights': self.scale_weights,
                      'data_format': self.data_format,
		      'force_fp32_output':self.force_fp32_output,
		      'fuse_relu': self.fuse_relu
                      }  
    def test_check_output(self):
        self.check_output()

    def init_bias(self):
        self.with_bias = False

    def init_data_type(self):
        self.srctype = np.int8
        self.dsttype = np.uint8 if self.fuse_relu else np.int8
        self.dsttype = np.float32 if self.force_fp32_output else self.dsttype

    def init_fuse_relu(self):
        self.fuse_relu = True

    def init_force_fp32_output(self):
        self.force_fp32_output = False

    def init_src_dims(self):
	self.src_dim = 4
 
    def init_data(self):
        if self.src_dim == 2:
		self.matrix = MatrixGenerate(5, 3, 1, 1, 1, self.srctype, True)
        else:
		self.matrix = MatrixGenerate(5, 3, 1, 5, 5, self.srctype)

    def test_check_grad_normal(self):
        pass

    def test_check_grad_no_weight(self):
        pass

 
class TestFCMKLDNNOp1x1(TestFCMKLDNNOp):
    def init_data(self):
       if self.src_dim == 2:
                self.matrix = MatrixGenerate(5, 3, 1, 1, 1, self.srctype, True)
       else:
                self.matrix = MatrixGenerate(10, 300, 1, 1, 1, self.srctype)
   
class TestFCMKLDNNOpBias(TestFCMKLDNNOp):
    def init_bias(self):
        self.with_bias = True

class TestFCMKLDNNOpSrcDims2(TestFCMKLDNNOp):
    def init_src_dims(self):
        self.src_dim = 2

def init_data_type_with_fusion(self, input_dt, fuse_relu, force_fp32_output):
    self.fuse_relu = fuse_relu
    self.force_fp32_output = force_fp32_output
    self.srctype = input_dt
    self.dsttype = np.uint8 if fuse_relu else np.int8
    self.dsttype = np.float32 if force_fp32_output else self.dsttype

def create_test_int8_class(parent):
     #--------------------test FC s8 in and u8 out--------------------

    class TestS8U8Case(parent):
        def init_data_type(self):
	    
            init_data_type_with_fusion(self, np.int8, True, False)

    #--------------------test FC s8 in and s8 out--------------------

    class TestS8S8Case(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.int8, False, False)

    #--------------------test FC u8 in and s8 out--------------------

    class TestU8S8Case(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.uint8, False, False)

    #--------------------test FC u8 in and u8 out--------------------

    class TestU8U8Case(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.uint8, True, False)

    #--------------------test FC u8 in and FP32 out With fuse_relu --------------------

    class TestU8FP32CaseWithRelu(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.uint8, True, True)

    #--------------------test FC s8 in and FP32 out With fuse_relu --------------------
    class TestS8FP32CaseWithRelu(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.int8, True, True)

    #--------------------test FC s8 in and FP32 out NO fuse_relu -------------------- 
    class TestS8FP32CaseNoRelu(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.int8, False, True)

    #--------------------test FC u8 in and FP32 out NOfuse_relu  --------------------
    class TestU8FP32CaseNoRelu(parent):
        def init_data_type(self):
            init_data_type_with_fusion(self, np.uint8, False, True)
    
    cls_name_s8u8 = "{0}_relu_{1}".format(parent.__name__, "1")
    cls_name_s8s8 = "{0}_relu_{1}".format(parent.__name__, "0")
    cls_name_u8s8 = "{0}_relu_{1}".format(parent.__name__, "0")
    cls_name_u8u8 = "{0}_relu_{1}".format(parent.__name__, "1")
    cls_name_u8fp_relu = "{0}_force_fp32_output_{1}_relu_{2}".format(parent.__name__, "1", "1")
    cls_name_u8fp = "{0}_force_fp32_output_{1}_relu_{2}".format(parent.__name__, "1", "0")
    cls_name_s8fp_relu = "{0}_force_fp32_output_{1}_relu{2}".format(parent.__name__, "1", "1")
    cls_name_s8fp = "{0}_force_fp32_output_{1}_relu{2}".format(parent.__name__, "1", "0")
    TestS8U8Case.__name__ = cls_name_s8u8
    TestS8S8Case.__name__ = cls_name_s8s8
    TestU8S8Case.__name__ = cls_name_u8s8
    TestU8U8Case.__name__ = cls_name_u8u8
    TestU8FP32CaseNoRelu.__namme__ = cls_name_u8fp
    TestU8FP32CaseWithRelu.__name__ = cls_name_u8fp_relu
    TestS8FP32CaseNoRelu.__name__ = cls_name_s8fp
    TestS8FP32CaseWithRelu._name__ = cls_name_s8fp_relu
    globals()[cls_name_s8u8] = TestS8U8Case
    globals()[cls_name_s8s8] = TestS8S8Case
    globals()[cls_name_u8s8] = TestU8S8Case
    globals()[cls_name_u8u8] = TestU8U8Case
    globals()[cls_name_u8fp] = TestU8FP32CaseNoRelu
    globals()[cls_name_u8fp_relu] = TestU8FP32CaseWithRelu
   # globals()[cls_name_s8fp] = TestS8FP32CaseNoRelu
    #globals()[cls_name_s8fp_relu] = TestS8FP32CaseWithRelu

create_test_int8_class(TestFCMKLDNNOp)
create_test_int8_class(TestFCMKLDNNOp1x1)
#create_test_int8_class(TestFCMKLDNNOpBias)
create_test_int8_class(TestFCMKLDNNOpSrcDims2)

if __name__ == "__main__":
    unittest.main()
