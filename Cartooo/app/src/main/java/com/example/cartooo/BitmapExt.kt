package com.example.cartooo

import android.graphics.Bitmap
import java.nio.ByteBuffer

/**
 * Developed by
 * @author Elad Finish
 */


fun Bitmap.toByteBuffer(): ByteBuffer {

    //calculate how many bytes our image consists of.
    val bytes: Int = this.byteCount
    //or we can calculate bytes this way. Use a different value than 4 if you don't use 32bit images.
    //int bytes = b.getWidth()*b.getHeight()*4;

    //or we can calculate bytes this way. Use a different value than 4 if you don't use 32bit images.
    //int bytes = b.getWidth()*b.getHeight()*4;

    return ByteBuffer.allocate(bytes) //Create a new buffer

//    this.copyPixelsToBuffer(buffer) //Move the byte data to the buffer
//
//    //Get the underlying array containing the data.
//    return buffer.array()
}
