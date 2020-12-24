package com.example.cartooo

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.util.Log
import java.nio.ByteBuffer

/**
 * Developed by
 * @author Elad Finish
 */

fun ByteBuffer.toBitmap(): Bitmap {
    val imageBytes = ByteArray(this.remaining())
    this.get(imageBytes)
    Log.d("FloatArray__1", "imageBytes " + imageBytes.contentToString())
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
}

