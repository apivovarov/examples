package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;

import java.io.IOException;

public class DLRKerasMobileNetV2 extends DLRModelBase {

    protected static long[] inShape = new long[] {1,3,224,224};

    public DLRKerasMobileNetV2(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected long[] getInShape() {
        return inShape;
    }

    @Override
    protected boolean isNCHW() {
        return true;
    }

    @Override
    protected String getModelPath() {
        return "dlr_keras_mobilenet_v2";
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        imgData.putFloat((((pixelValue >> 16) & 0xFF) / 127.5f) - 1.0f);
        imgData.putFloat((((pixelValue >> 8) & 0xFF) / 127.5f) - 1.0f);
        imgData.putFloat(((pixelValue & 0xFF) / 127.5f) - 1.0f);
    }

    @Override
    protected String getLabelPath() {
        return "labels1000.txt";
    }
}
