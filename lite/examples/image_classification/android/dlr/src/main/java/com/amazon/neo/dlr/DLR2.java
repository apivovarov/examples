package com.amazon.neo.dlr;

public class DLR2 {

    static {
        System.loadLibrary("dlr2");
    }

    public static int getFive() {
        return 5;
    }

    public static double getV(double v) {
        return v;
    }

    public static native double passDouble(double i);

    public static native int passInt(int i);

    public static native String getHello();

    public static native long createHandle();

    public static native String useHandle(long handle);

    public static native float fillArr(float[] arr);
}
