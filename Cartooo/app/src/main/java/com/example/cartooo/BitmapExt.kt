package com.example.cartooo

import android.graphics.Bitmap
import java.nio.ByteBuffer

/**
 * Developed by
 * @author Elad Finish
 */


fun Bitmap.toByteBuffer(): ByteBuffer {

    //calculate how many bytes our image consists of.
    val bytes = this.byteCount
    //or we can calculate bytes this way. Use a different value than 4 if you don't use 32bit images.
    //int bytes = b.getWidth()*b.getHeight()*4;

    //or we can calculate bytes this way. Use a different value than 4 if you don't use 32bit images.
    //int bytes = b.getWidth()*b.getHeight()*4;

    val buffer = ByteBuffer.allocate(bytes) //Create a new buffer

    this.copyPixelsToBuffer(buffer) //Move the byte data to the buffer

    return buffer

//    //Get the underlying array containing the data.
//    return buffer.array()
}


// For input image where information is only stored in alpha channel over black RGB.
// Make all pixels with transparency black, and remove alpha channel.
fun Bitmap.alphaToBlack(): Bitmap {
    val rgbImage = this.copy(Bitmap.Config.ARGB_8888, true)
    for (y in 0 until rgbImage.height) {
        for (x in 0 until rgbImage.width) {
//            val aPixel = rgbImage.getPixel(x, y)
            if (rgbImage.getPixel(x, y) < -0x1000000) rgbImage.setPixel(x, y, -0x1000000)
        }
    }
    return rgbImage
}

fun Bitmap.onlyRGB(): Bitmap {
    return this.copy(Bitmap.Config.ALPHA_8, true)
}


