package com.example.tomatocure_bug_fix

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import coil.compose.rememberAsyncImagePainter
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : ComponentActivity() {

    // Declare the ActivityResultLaunchers for both camera and gallery
    private lateinit var cameraResultLauncher: ActivityResultLauncher<Uri>
    private lateinit var galleryResultLauncher: ActivityResultLauncher<String>
    private lateinit var tflite: Interpreter
    private lateinit var labels: List<String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Load the TFLite model and labels
        tflite = Interpreter(loadModelFile())
        labels = assets.open("labels.txt").bufferedReader().readLines()

        // Create a URI for saving the image from the camera
        val imageUri: Uri = createImageUri()

        // State to hold the selected image URI
        val selectedImageUri = mutableStateOf<Uri?>(null)

        // Initialize the ActivityResultLauncher for the camera
        cameraResultLauncher = registerForActivityResult(ActivityResultContracts.TakePicture()) { isSuccess: Boolean ->
            if (isSuccess) {
                println("Image captured successfully: $imageUri")
                selectedImageUri.value = imageUri // Update the selected image URI
            } else {
                println("Image capture failed")
            }
        }

        // Initialize the ActivityResultLauncher for selecting an image from the gallery
        galleryResultLauncher = registerForActivityResult(ActivityResultContracts.GetContent()) { uri: Uri? ->
            uri?.let {
                println("Selected image from gallery: $it")
                selectedImageUri.value = it // Update the selected image URI
            } ?: run {
                println("No image selected from gallery")
            }
        }

        setContent {
            MaterialTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    MainScreen(cameraResultLauncher, galleryResultLauncher, imageUri, selectedImageUri)
                }
            }
        }
    }

    // Function to create a URI to store the image
    private fun createImageUri(): Uri {
        val contentValues = ContentValues().apply {
            put(MediaStore.Images.Media.TITLE, "Captured Image")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
        }
        return contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, contentValues)!!
    }

    // Function to load the TFLite model
    private fun loadModelFile(): MappedByteBuffer {
        val fileDescriptor = assets.openFd("model(1).tflite") // Make sure the model file is named "model.tflite"
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    // Function to preprocess the image
    private fun preprocessImage(uri: Uri, modelInputSize: Int): ByteBuffer {
        val inputStream = contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        val scaledBitmap = Bitmap.createScaledBitmap(bitmap, modelInputSize, modelInputSize, true)
        val inputBuffer = ByteBuffer.allocateDirect(4 * modelInputSize * modelInputSize * 3)
        inputBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(modelInputSize * modelInputSize)
        scaledBitmap.getPixels(intValues, 0, scaledBitmap.width, 0, 0, scaledBitmap.width, scaledBitmap.height)
        for (pixelValue in intValues) {
            inputBuffer.putFloat((pixelValue shr 16 and 0xFF) / 255.0f) // Red
            inputBuffer.putFloat((pixelValue shr 8 and 0xFF) / 255.0f)  // Green
            inputBuffer.putFloat((pixelValue and 0xFF) / 255.0f)       // Blue
        }
        return inputBuffer
    }

    @Composable
    fun MainScreen(
        cameraResultLauncher: ActivityResultLauncher<Uri>,
        galleryResultLauncher: ActivityResultLauncher<String>,
        imageUri: Uri,
        selectedImageUri: MutableState<Uri?>
    ) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFFE1BEE7)) // Set a light purple background color
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Top
        ) {
            Greeting(name = "TomatoCure")

            Spacer(modifier = Modifier.height(16.dp))

            // Display the selected image or placeholder
            if (selectedImageUri.value != null) {
                Image(
                    painter = rememberAsyncImagePainter(model = selectedImageUri.value),
                    contentDescription = "Selected Image",
                    modifier = Modifier
                        .size(200.dp)
                        .padding(8.dp),
                    contentScale = ContentScale.Crop
                )
            } else {
                Image(
                    painter = painterResource(id = R.drawable.camera_image), // Placeholder image
                    contentDescription = "Placeholder Image",
                    modifier = Modifier
                        .size(200.dp)
                        .padding(8.dp)
                )
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = { cameraResultLauncher.launch(imageUri) }) {
                Text(text = "Capture a Picture")
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = { galleryResultLauncher.launch("image/*") }) {
                Text(text = "Open Gallery")
            }

            Spacer(modifier = Modifier.height(16.dp))

            Button(onClick = {
                selectedImageUri.value?.let {
                    val inputBuffer = preprocessImage(it, modelInputSize = 224)
                    val output = Array(1) { FloatArray(labels.size) }
                    tflite.run(inputBuffer, output)

                    val predictedIndex = output[0].indices.maxByOrNull { output[0][it] } ?: -1
                    val diseaseName = if (predictedIndex >= 0) labels[predictedIndex] else "Unknown"
                    println("Detected Disease: $diseaseName")
                } ?: run {
                    println("No image selected")
                }
            }) {
                Text(text = "Detect")
            }
        }
    }

    @Composable
    fun Greeting(name: String, modifier: Modifier = Modifier) {
        Text(
            text = " $name",
            style = MaterialTheme.typography.titleLarge.copy(fontWeight = FontWeight.Bold),
            color = Color.Black, // Change text to black
            modifier = modifier
        )
    }

    @Preview(showBackground = true)
    @Composable
    fun GreetingPreview() {
        MaterialTheme {
            val mockUri = remember { mutableStateOf<Uri?>(null) }
            MainScreen(cameraResultLauncher, galleryResultLauncher, Uri.EMPTY, mockUri)  // Empty Uri in preview
        }
    }
}
