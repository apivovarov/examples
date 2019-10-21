package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;

import java.io.IOException;

public class DLRGluonCVResNet18 extends DLRModelBase {

    protected static long[] inShape = new long[] {1,3,224,224};

    public DLRGluonCVResNet18(Activity activity) throws IOException {
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
        return "dlr_gluoncv_resnet18_v2";
    }

    @Override
    protected void addPixelValue(int pixelValue) {
        int r = ((pixelValue >> 16) & 0xFF);
        int g = ((pixelValue >> 8) & 0xFF);
        int b = (pixelValue & 0xFF);
        float rf = (r - 123) / 58.395f;
        float gf = (g - 117) / 57.12f;
        float bf = (b - 104) / 57.375f;
        //Log.i("DLR", "(" + rf + ","+ gf + ","+ bf + ")");
        imgData.putFloat(rf);
        imgData.putFloat(gf);
        imgData.putFloat(bf);
    }

    @Override
    protected String getLabelPath() {
        return "labels1000.txt";
    }
}
