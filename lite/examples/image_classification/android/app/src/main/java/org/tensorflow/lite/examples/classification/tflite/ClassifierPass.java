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
import android.util.Log;

import java.io.*;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.Arrays;
import com.amazon.neo.dlr.DLR2;


/** This TensorFlowLite classifier works with the float MobileNet model. */
public class ClassifierPass extends Classifier {

  /**
   * An array to hold inference results, to be feed into Tensorflow Lite as outputs. This isn't part
   * of the super class, because we need a primitive array here.
   */
  private float[][] labelProbArray;

//  void printDir(File f) {
//    File[] files = f.listFiles();
//    for (File inFile : files) {
//      if (inFile.isDirectory()) {
//        Log.i("DLR", "- dir: " + inFile.toString());
//        printDir(inFile);
//      } else {
//        Log.i("DLR", "- file: " + inFile.toString());
//      }
//    }
//  }

  private File createFileFromInputStream(InputStream inputStream, String filePath) {

    try{
      File f = new File(filePath);
      OutputStream outputStream = new FileOutputStream(f);
      byte buffer[] = new byte[1024];
      int length = 0;

      while((length=inputStream.read(buffer)) > 0) {
        outputStream.write(buffer,0,length);
      }

      outputStream.close();
      inputStream.close();

      return f;
    }catch (IOException e) {
      //Logging exception
    }

    return null;
  }

  /**
   * Initializes a {@code ClassifierFloatMobileNet}.
   *
   * @param activity
   */
  public ClassifierPass(Activity activity, Device device, int numThreads)
      throws IOException {
    super(activity);
    labelProbArray = new float[1][getNumLabels()];
  }

  @Override
  public int getImageSizeX() {
    return 224;
  }

  @Override
  public int getImageSizeY() {
    return 224;
  }

  protected MappedByteBuffer loadModelFile(Activity activity) throws IOException {
    return null;
  }
  @Override
  protected String getModelPath() {
    // you can download this file from
    // see build.gradle for where to obtain this file. It should be auto
    // downloaded into assets.
    return "/data/user/0/org.tensorflow.lite.examples.classification/dlr_mobilenet_v1_1.0_224";
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
    //pass
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

//    long h = DLR2.createHandle();
//    String s = DLR2.useHandle(h);
//    Log.i("DLR2", s == null ? "NULL" : s);
//    float[] arr = new float[10];
//    arr[0]=22.2f;
//    float res = DLR2.fillArr(arr);
//    Log.i("res", "res: " + res);
//    Log.i("arr", "arr0: " + arr[0]);
//    Log.i("arr", "arr1: " + arr[1]);
//    Log.i("arr", "arr2: " + arr[2]);
//    Log.i("arr", "arr4: " + arr[4]);
    double v = DLR2.passDouble(0.5);
    Arrays.fill(labelProbArray[0], 0.0f);
    if (Math.random() > v) {
      labelProbArray[0][284] = 0.998f;
      labelProbArray[0][285] = 0.61f;
      labelProbArray[0][286] = 0.51f;
    } else {
      labelProbArray[0][153] = 0.998f;
      labelProbArray[0][154] = 0.61f;
      labelProbArray[0][155] = 0.51f;
    }
  }
}
