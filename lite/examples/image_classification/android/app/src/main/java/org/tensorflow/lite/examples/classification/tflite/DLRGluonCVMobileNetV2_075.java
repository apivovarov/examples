package org.tensorflow.lite.examples.classification.tflite;

import android.app.Activity;

import java.io.IOException;

public class DLRGluonCVMobileNetV2_075 extends DLRGluonCVResNet18 {

    public DLRGluonCVMobileNetV2_075(Activity activity) throws IOException {
        super(activity);
    }

    @Override
    protected String getModelPath() {
        return "dlr_gluoncv_mobilenet_v2_075";
    }
}
