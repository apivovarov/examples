/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;
import android.content.res.AssetManager;
import android.util.Log;

import com.amazon.neo.dlr.DLR;
import static org.tensorflow.lite.examples.classification.env.ImageUtils.createFileFromInputStream;

import java.io.*;
import java.util.Arrays;

/** This TensorFlowLite classifier works with the float MobileNet model. */
public class DLRFloatMobileNet extends Classifier {

  /** MobileNet requires additional normalization of the used input. */
  private static final float IMAGE_MEAN = 127.5f;
  private static final float IMAGE_STD = 127.5f;

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][] labelProbArray = null;

  protected long handle;

  long[] inShape = new long[] {1,224,224,3};
  int sz = 3*224*224;
  float[] input = new float[sz];

  /**
   * Initializes a {@code ClassifierFloatMobileNet}.
   *
   * @param activity
   */
  public DLRFloatMobileNet(Activity activity, Device device, int numThreads)
      throws IOException {
    super(activity);
    labelProbArray = new float[1][getNumLabels()];

    AssetManager am = activity.getAssets();

    File dd = new File(activity.getApplicationContext().getApplicationInfo().dataDir,
            getModelPath());
    String fullModelPath = dd.toString();
    dd.mkdir();

    InputStream inputStream = am.open(getModelPath() + "/model.so");
    createFileFromInputStream(inputStream, fullModelPath + "/model.so");
    inputStream.close();
    inputStream = am.open(getModelPath() + "/model.json");
    createFileFromInputStream(inputStream, fullModelPath + "/model.json");
    inputStream.close();
    inputStream = am.open(getModelPath() + "/model.params");
    createFileFromInputStream(inputStream, fullModelPath + "/model.params");
    inputStream.close();

    //File f = new File(activity.getApplicationContext().getApplicationInfo().dataDir);

    //printDir(f);
    labelProbArray = new float[1][getNumLabels()];
    handle = DLR.CreateDLRModel(fullModelPath, 1, 0);
    Log.i("DLR", "CreateDLRModel: " + handle);
    if (handle == 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("CreateDLRModel failed");
    }
    Log.i("DLR", "GetDLRBackend: " + DLR.GetDLRBackend(handle));
    Log.i("DLR", "GetDLRNumInputs: " + DLR.GetDLRNumInputs(handle));
    Log.i("DLR", "GetDLRNumWeights: " + DLR.GetDLRNumWeights(handle));
    Log.i("DLR", "GetDLRNumOutputs: " + DLR.GetDLRNumOutputs(handle));

    Log.i("DLR", "GetDLRInputName[0]: " + DLR.GetDLRInputName(handle, 0));
    Log.i("DLR", "GetDLRWeightName[4]: " + DLR.GetDLRWeightName(handle, 4));
    // GetDLROutputSize and Dim
    int outDim = DLR.GetDLROutputDim(handle, 0);
    long outSize = DLR.GetDLROutputSize(handle, 0);
    Log.i("DLR", "GetDLROutputSize[0]: " + outSize);
    Log.i("DLR", "GetDLROutputDim[0]: " + outDim);
    //GetDLROutputShape
    long[] out_shape = new long[outDim];
    if (DLR.GetDLROutputShape(handle, 0, out_shape) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("GetDLROutputShape failed");
    }
    Log.i("DLR", "GetDLROutputShape[0]: " + Arrays.toString(out_shape));
  }

  @Override
  public int getImageSizeX() {
    return 224;
  }

  @Override
  public int getImageSizeY() {
    return 224;
  }

  @Override
  protected String getModelPath() {
    // you can download this file from
    // see build.gradle for where to obtain this file. It should be auto
    // downloaded into assets.
    return "dlr_mobilenet_v1_1.0_224";
  }

  @Override
  protected String getLabelPath() {
    return "labels.txt";
  }

  @Override
  protected int getNumBytesPerChannel() {
    return 4; // Float.SIZE / Byte.SIZE;
  }

  @Override
  protected void addPixelValue(int pixelValue) {
    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
  }

  @Override
  protected float getProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void setProbability(int labelIndex, Number value) {
    labelProbArray[0][labelIndex] = value.floatValue();
  }

  @Override
  protected float getNormalizedProbability(int labelIndex) {
    return labelProbArray[0][labelIndex];
  }

  @Override
  protected void runInference() {
    //Log.i("DLR", "img.HasArray: " + imgData.hasArray() + ", len: " + imgData.array().length);
    imgData.rewind();
    imgData.asFloatBuffer().get(input, 0, sz);
    if (DLR.SetDLRInput(handle, "input", inShape, input, 4) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("SetDLRInput failed");
    }
    //Log.i("DLR", "SetDLRInput: OK");

    if (DLR.RunDLRModel(handle) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("RunDLRModel failed");
    }
    //Log.i("DLR", "RunDLRModel: OK");
    float[] output = labelProbArray[0];
    if (DLR.GetDLROutput(handle, 0, output) != 0) {
      Log.i("DLR", "DLRGetLastError: " + DLR.DLRGetLastError());
      throw new RuntimeException("GetDLROutput failed");
    }
  }

  @Override
  public void close() {
    super.close();
    if (handle > 0) {
      DLR.DeleteDLRModel(handle);
      handle = 0;
    }
  }
}
