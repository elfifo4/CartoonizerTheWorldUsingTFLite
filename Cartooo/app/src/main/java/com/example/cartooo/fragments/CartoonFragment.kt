package com.example.cartooo.fragments

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.view.*
import android.widget.Toast
import androidx.core.graphics.drawable.toBitmap
import androidx.fragment.app.Fragment
import androidx.navigation.fragment.navArgs
import com.bumptech.glide.Glide
import com.example.cartooo.*
import com.example.cartooo.ml.*
import kotlinx.android.synthetic.main.fragment_selfie2cartoon.*
import kotlinx.coroutines.*
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.model.Model
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.text.SimpleDateFormat
import java.util.*


//展示捕获的输入图像和由tflite模型卡通化的输出图像
class CartoonFragment : Fragment() {

    private val TAG = this.javaClass.simpleName

    private val args: CartoonFragmentArgs by navArgs()
    private lateinit var filePath: String
    private var modelType: Int = 0

    private val parentJob = Job()

    //    协程
    private val coroutineScope = CoroutineScope(
        Dispatchers.Main + parentJob
    )

    private fun getOutputAsync(bitmap: Bitmap): Deferred<Pair<Bitmap, Long>> =
//使用async（）在IO优化的分派器中创建协程以进行模型推断
        coroutineScope.async(Dispatchers.IO) {
//GPU代理
            val option = Model.Options.Builder()
                .setDevice(Model.Device.GPU)
                .setNumThreads(4)
                .build()
//        输入
            val sourceImage = TensorImage.fromBitmap(bitmap)

//        输出
            var cartoonizedImage: TensorImage? = null
            var res: Bitmap? = null

            val startTime = SystemClock.uptimeMillis()
            when (modelType) {
                0 -> res = inferenceWithVedantaNewModel32(bitmap) // DR Vedanta
                1 -> cartoonizedImage = inferenceWithDrModel(sourceImage)       //DR
                2 -> cartoonizedImage = inferenceWithFp16Model(sourceImage)     //Fp16
                3 -> cartoonizedImage = inferenceWithInt8Model(sourceImage, option) //Int8
                4 -> res = inferenceWithHayaoModel(bitmap) //Hayao
                else -> cartoonizedImage = inferenceWithDrModel(sourceImage)
                //        此推断时间包括预处理和后处理

            }
            val inferenceTime = SystemClock.uptimeMillis() - startTime

            if (cartoonizedImage != null) {
                val cartoonizedImageBitmap = cartoonizedImage.bitmap
                return@async Pair(cartoonizedImageBitmap, inferenceTime)
            } else {
                return@async Pair(res!!, inferenceTime)
            }


        }

    private fun inferenceWithVedantaNewModel32(bitmap: Bitmap): Bitmap {

        val model = WhiteboxCartoon32.newInstance(requireContext())

        val scaledImage = Bitmap.createScaledBitmap(bitmap, DESIRED_SIZE, DESIRED_SIZE, true)

        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 720, 720, 3), DataType.FLOAT32)


        val inputImageBuffer = TensorImage(DataType.FLOAT32)
        inputImageBuffer.load(scaledImage)
        inputFeature0.loadBuffer(inputImageBuffer.buffer)


        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        val result: FloatArray = outputFeature0.floatArray
        Log.d(TAG, "FloatArray__1 floatArray" + result.contentToString())
        Log.d(TAG, "FloatArray__1 intArray" + outputFeature0.intArray.contentToString())

        val outputBitmap: Bitmap = postProcess(result)

        model.close()

//        return outputFeature0.buffer.toBitmap()
        return outputBitmap
    }


    private fun inferenceWithVedantaNewModel2(bitmap: Bitmap): Bitmap {

        val model = WhiteboxCartoonUint8.newInstance(requireContext())

        val scaledImage = Bitmap.createScaledBitmap(bitmap, DESIRED_SIZE, DESIRED_SIZE, true)

        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 720, 720, 3), DataType.UINT8)


        val inputImageBuffer = TensorImage(DataType.UINT8)
        inputImageBuffer.load(scaledImage)
        inputFeature0.loadBuffer(inputImageBuffer.buffer)


        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

//        TFImageUtils.convertBitmapToTensorBuffer(
//            scaled,
//            tensorBuffer
//        )


        val result: FloatArray = outputFeature0.floatArray
        Log.d(TAG, "FloatArray__2 " + result.contentToString())

        val outputBitmap: Bitmap = postProcess(result)


//        val outputBitmap = TFImageUtils.convertTensorBufferToBitmap(outputFeature0)

        model.close()

        return outputBitmap
    }

    private fun postProcess(data: FloatArray): Bitmap {
        val width = DESIRED_SIZE
        val height = DESIRED_SIZE
        val result = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until width) {
            for (x in 0 until height) {
                val pos = y * width * 3 + x * 3
                val color = rgb(
                    data[pos],
                    data[pos + 1],
                    data[pos + 2]
                )
                result.setPixel(x, y, color)
            }
        }
        return result
    }

    private fun conv(value: Float): Int {
        return (255.0 * (value + 1f) / 2f).toInt()
    }

    private fun rgb(red: Float, green: Float, blue: Float): Int {
        return -0x1000000 or (conv(red) shl 16) or (conv(green) shl 8) or conv(blue)
    }

    private fun inferenceWithVedantaNewModel(bitmap: Bitmap): Bitmap {
        Log.d(TAG, "inferenceWithDrVedantaModel")

        val model = WhiteboxCartoonUint8.newInstance(requireContext())


        inImgData.clear()
        outImgData.clear()

        val scaled = Bitmap.createScaledBitmap(bitmap, DESIRED_SIZE, DESIRED_SIZE, true)
        convertBitmapToByteBuffer(scaled)

        Log.d(TAG, "inImgData___ $inImgData")

        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 720, 720, 3), DataType.UINT8)
        inputFeature0.loadBuffer(inImgData)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        model.close()


        val output = scaled.copy(Bitmap.Config.ARGB_8888, true)

        convertOutputBufferToBitmap(outputFeature0.buffer, output)
        return scaled
    }


    private fun inferenceWithDrVedantaModel(bitmap: Bitmap): Bitmap {
        Log.d(TAG, "inferenceWithDrVedantaModel")

        val model: WhiteboxCartoonUint8 = WhiteboxCartoonUint8.newInstance(requireContext())

        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(intArrayOf(1, 720, 720, 3), DataType.UINT8)

        inImgData.clear()
        outImgData.clear()

        val scaled = Bitmap.createScaledBitmap(bitmap, DESIRED_SIZE, DESIRED_SIZE, true)
        convertBitmapToByteBuffer(scaled)

//        Log.d(TAG, "inImgData___ $inImgData")
//        Log.d(TAG, "byteBuffer___ $byteBuffer")

        inputFeature0.loadBuffer(inImgData)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0: TensorBuffer = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        model.close()


        val output = scaled.copy(Bitmap.Config.ARGB_8888, true)

        convertOutputBufferToBitmap(outputFeature0.buffer, output)


//        ByteBuffer
//        ImageUtilsNew.convertArrayToBitmap()
        return output
    }

    private fun inferenceWithHayaoModel(bitmap: Bitmap): Bitmap {
        Log.d(TAG, "inferenceWithHayaoModel")

        val model = Hayao.newInstance(requireContext())

        // Creates inputs for reference.
        val inputFeature0 =
            TensorBuffer.createFixedSize(
                intArrayOf(1, 720, 720, 3),
                DataType.FLOAT32
            ) //DataType.UINT8

        inImgData.clear()
        outImgData.clear()

        val scaled = Bitmap.createScaledBitmap(bitmap, DESIRED_SIZE, DESIRED_SIZE, true)
        convertBitmapToByteBuffer(scaled)

        Log.d(TAG, "inImgData___ $inImgData")

        inputFeature0.loadBuffer(inImgData)

        // Runs model inference and gets result.
        val outputs = model.process(inputFeature0)
        val outputFeature0 = outputs.outputFeature0AsTensorBuffer

        // Releases model resources if no longer used.
        model.close()

        val output = scaled.copy(Bitmap.Config.ARGB_8888, true)

        convertOutputBufferToBitmap(outputFeature0.buffer, output)
        return output
    }

    //使用动态范围tflite模型进行推断
    private fun inferenceWithDrModel(sourceImage: TensorImage): TensorImage {

        Log.d(TAG, "inferenceWithDrModel")
        val model = WhiteboxCartoonGanDr.newInstance(requireContext())

//    运行模型推断并获取结果。
        val outputs = model.process(sourceImage)
        val cartoonizedImage = outputs.cartoonizedImageAsTensorImage

//如果不再使用，则释放模型资源。
        model.close()

        return cartoonizedImage
    }

    //    使用fp16 tflite模型进行推断
    private fun inferenceWithFp16Model(sourceImage: TensorImage): TensorImage {
        Log.d(TAG, "inferenceWithFp16Model")

        val model = WhiteboxCartoonGanFp16.newInstance(requireContext())

        //    运行模型推断并获取结果。
        val outputs = model.process(sourceImage)
        val cartoonizedImage = outputs.cartoonizedImageAsTensorImage

        //如果不再使用，则释放模型资源。
        model.close()

        return cartoonizedImage
    }

    //    使用int8 tflite模型进行推断
    private fun inferenceWithInt8Model(
        sourceImage: TensorImage,
        options: Model.Options
    ): TensorImage {
        Log.d(TAG, "inferenceWithInt8Model")

        val model = WhiteboxCartoonGanInt8.newInstance(requireContext(), options)

        //    运行模型推断并获取结果。
        val outputs = model.process(sourceImage)
        val cartoonizedImage = outputs.cartoonizedImageAsTensorImage

        //如果不再使用，则释放模型资源。
        model.close()

        return cartoonizedImage
    }

    private fun updateUI(outputBitmap: Bitmap, inferenceTime: Long) {
        prograssbar.visibility = View.GONE
        imageview_output?.setImageBitmap(outputBitmap)
        inference_info.setText("推断时间: " + inferenceTime.toString() + "ms")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)


        setHasOptionsMenu(true) //启用工具栏

        retainInstance = true  //保留实例
        filePath = args.rootDir
        modelType = args.modelType

    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View? {
//        填充此片段的布局
        return inflater.inflate(R.layout.fragment_selfie2cartoon, container, false)
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        val photoFile = File(filePath)

        Glide.with(imageview_input.context)
            .load(photoFile)
            .into(imageview_input)

        val selfieBitmap = BitmapFactory.decodeFile(filePath)
        coroutineScope.launch(Dispatchers.Main) {
            val (outputBitmap, inferenceTime) = getOutputAsync(selfieBitmap).await()
            updateUI(outputBitmap, inferenceTime)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
//清理协程任务
        parentJob.cancel()
    }

    override fun onCreateOptionsMenu(menu: Menu, inflater: MenuInflater) {
        inflater.inflate(R.menu.menu_main, menu)
        super.onCreateOptionsMenu(menu, inflater)
    }

    override fun onOptionsItemSelected(item: MenuItem): Boolean {
        when (item.itemId) {
            R.id.action_save -> saveCartoon()
        }
        return super.onOptionsItemSelected(item)
    }

    private fun saveCartoon(): String {
        val cartoonBitmap = imageview_output.drawable.toBitmap()
        val file = File(
            MainActivity.getOutputDirectory(requireContext()),
            SimpleDateFormat(
                FILENAME_FORMAT, Locale.CHINA
            ).format(System.currentTimeMillis()) + "_cartoon.jgp"
        )
        ImageUtils.saveBitmap(cartoonBitmap, file)
        Toast.makeText(context, "已被保存至" + file.absolutePath.toString(), Toast.LENGTH_SHORT).show()

        return file.absolutePath
    }

    companion object {
        private const val TAG = "CartoonFragment"
        private const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"

        private const val DESIRED_SIZE = 720

    }


    private val mapSize by lazy(LazyThreadSafetyMode.NONE) { DESIRED_SIZE * DESIRED_SIZE }
    private val pixelsBuffer = IntArray(mapSize)
    private val inImgData: ByteBuffer =
        ByteBuffer.allocateDirect(mapSize * 3 * 8 / java.lang.Byte.SIZE)
            .apply { order(ByteOrder.nativeOrder()) }
    private val outImgData: ByteBuffer =
        ByteBuffer.allocateDirect(mapSize * java.lang.Float.SIZE / java.lang.Byte.SIZE)
            .apply { order(ByteOrder.nativeOrder()) }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap) {
        bitmap.getPixels(pixelsBuffer, 0, DESIRED_SIZE, 0, 0, DESIRED_SIZE, DESIRED_SIZE)
        inImgData.rewind()
        // Convert the image to floating point.
        var pixel = 0
        for (i in 0 until DESIRED_SIZE) {
            for (j in 0 until DESIRED_SIZE) {
                val pixelValue = pixelsBuffer[pixel++]
                val v1 = (pixelValue shr 16 and 0xFF) / 255f
                val v2 = (pixelValue shr 8 and 0xFF) / 255f
                val v3 = (pixelValue and 0xFF) / 255f
                inImgData.putFloat(v1)
                inImgData.putFloat(v2)
                inImgData.putFloat(v3)
            }
        }
    }

    private fun convertOutputBufferToBitmap(outImgData: ByteBuffer, outBitmap: Bitmap) {
        outImgData.rewind()

        for (i in 0 until mapSize) {
            val value = outImgData.float
            if (value > 0.2) {
                pixelsBuffer[i] = -0x1
            } else {
                pixelsBuffer[i] = -0x1000000
            }
        }

        outBitmap.setPixels(pixelsBuffer, 0, DESIRED_SIZE, 0, 0, DESIRED_SIZE, DESIRED_SIZE)
    }

}