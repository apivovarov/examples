#include <jni.h>
#include <string>
#include <cstdint>

extern "C" JNIEXPORT jdouble JNICALL
Java_com_amazon_neo_dlr_DLR2_passDouble(JNIEnv* env, jobject thiz, jdouble i) {
    return i;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_amazon_neo_dlr_DLR2_passInt(JNIEnv* env, jobject thiz, jint i) {
    return i;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_amazon_neo_dlr_DLR2_getHello(JNIEnv* env, jobject thiz) {
    return env->NewStringUTF("Hello");
}

extern "C" JNIEXPORT jfloat JNICALL
Java_com_amazon_neo_dlr_DLR2_fillArr(JNIEnv* env, jobject thiz, jfloatArray arr) {
    jboolean isCopy = JNI_FALSE;
    jfloat* body = env->GetFloatArrayElements(arr, &isCopy);
    jfloat v = body[0];
    body[0] = 1.0f;
    body[1] = 1.1f;
    body[2] = 1.2f;
    body[3] = 1.3f;
    env->ReleaseFloatArrayElements(arr, body, 0);

    return v;
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_amazon_neo_dlr_DLR2_createHandle(JNIEnv* env, jobject thiz) {
    std::string* s = new std::string("foo");
    void* p = (void*)s;
    std::uintptr_t i = reinterpret_cast<std::uintptr_t>(p);
    return i;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_amazon_neo_dlr_DLR2_useHandle(JNIEnv* env, jobject thiz, jlong handle) {
    void* p = reinterpret_cast<void*>(handle);
    std::string* s = static_cast<std::string*>(p);
    //return env->NewStringUTF(s->c_str());
    return NULL;
}