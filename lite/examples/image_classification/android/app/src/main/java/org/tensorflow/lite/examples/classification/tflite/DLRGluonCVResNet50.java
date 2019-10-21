package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;

import java.io.IOException;

public class DLRGluonCVResNet50 extends DLRGluonCVResNet18 {

    public DLRGluonCVResNet50(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected String getModelPath() {
        return "dlr_gluoncv_resnet50_v2";
    }
}
