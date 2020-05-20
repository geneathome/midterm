#include "mbed.h"
#include <cmath>
#include "DA7212.h"

#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "uLCD_4DGL.h"

DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
Serial pc(USBTX, USBRX);

#define bufferLength (32)

uLCD_4DGL uLCD(D1, D0, D2);
InterruptIn button(SW2);
InterruptIn button1(SW3);
DigitalOut redLED(LED1);

int mode = 0;
int pause = 0;
char serialInBuffer[bufferLength];

Timer t;

void playNote(float freq)
{
  float frequency =  freq;
  //(int16_t) (freq[number])*((1<<16)-1) ;
  for (int i = 0; i < kAudioTxBufferSize; i++)
  {
  waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency /( 500*frequency))) * ((1<<16) - 1));
  }
  // the loop below will play the note for the duration of 1s
  for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
  {
    audio.spk.play(waveform, kAudioTxBufferSize);
  }

}

int serialCount =0;
float song_note[66];
void ISR2(){
  pause=!pause;
}
void ISR1(){
    if (mode == 1){
      mode = 3;
    } else if (mode == 5){
      mode = 6;
    } else {
      mode = 4;
    }
}

void loadSignal(int leng)
{
  int i = 0;
  serialCount = 0;
  audio.spk.pause();
  while(i < leng)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 5)
      {
        serialInBuffer[serialCount] = '\0';
        song_note[i] = (float) atof(serialInBuffer);
        serialCount = 0;
        i++;
        printf("%d", i);
      }
    }
    button.rise(&ISR1);
  }
}

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}



int main(int argc, char* argv[]) {

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction
  int gesture_index;
  int first = 0;

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                               tflite::ops::micro::Register_RESHAPE(),1);                             

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  uLCD.printf("\nMusic Player\n"); //Default Green on black text
  uLCD.printf("\n1: Little Star\n"); //Default Green on black text
  uLCD.printf("\n2: Birthday\n"); //Default Green on black text


  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    button.rise(&ISR1);

    if (mode == 3) {
      uLCD.cls();
      uLCD.printf("\nMusic Player\n"); //Default Green on black text
      uLCD.printf("\n1: Little Star\n"); //Default Green on black text
      uLCD.printf("\n2: Birthday\n"); //Default Green on black text
      mode = 0;
    } else if (mode == 4){
      uLCD.cls();
      uLCD.printf("\nSelect mode\n"); //Default Green on black text
      uLCD.printf("\n1: change song\n"); //Default Green on black text
      uLCD.printf("\n2: taiko\n"); //Default Green on black text
      mode = 2;
    } else if (mode == 6){
      uLCD.cls();
      uLCD.printf("\nTaiko mode\n"); //Default Green on black text
      uLCD.printf("\n1: Little Star\n"); //Default Green on black text
      uLCD.printf("\n2: Birthday\n"); //Default Green on black text
      mode = 5;
    }

    if (gesture_index < label_num) {
      // error_reporter->Report(config.output_message[gesture_index]);
      if (mode == 0) {
        if (first == 0){
          uLCD.printf("\nloading\n"); //Default Green on black text
          error_reporter->Report("0");
          redLED = 0;
          loadSignal(66);
          redLED = 1;
          first = 1;
        }
        uLCD.cls();
        uLCD.printf("\nPlaying song #%d\n", gesture_index+1); //Default Green on black text
        if(gesture_index == 0) {
          for (int j = 0; j < 42; j++) {
            if(pause==0){
            uLCD.printf("\nPlaying song #%f\n", 500*song_note[j]); //Default Green on black text
            playNote(song_note[j]);
            wait_us(1000000);
          } else
          {
            uLCD.cls();
            uLCD.printf("\npause");
            wait(1.0);
          }
          
            button1.rise(&ISR2);
            button.rise(&ISR1);
          }
        } else {
          for (int j = 42; j < 66; j++) {
            if(pause==0){
            uLCD.printf("\nPlaying song #%f\n", 500*song_note[j]); //Default Green on black text
            playNote(song_note[j]);
            wait_us(1000000);
            } else {
            uLCD.cls();
            uLCD.printf("\npause");
            wait(1.0);
            }
            button1.rise(&ISR2);
            button.rise(&ISR1);
          }
        }
      } else if (mode == 2) {
        if (gesture_index == 0) {
          uLCD.cls();
          uLCD.printf("\nMusic Player\n"); //Default Green on black text
          uLCD.printf("\n1: Little Star\n"); //Default Green on black text
          uLCD.printf("\n2: Birthday\n"); //Default Green on black text
          mode = 0;
        } else {
          uLCD.cls();
          uLCD.printf("\nTaiko mode\n"); //Default Green on black text
          uLCD.printf("\n1: Little Star\n"); //Default Green on black text
          uLCD.printf("\n2: Birthday\n"); //Default Green on black text
          mode = 5;
        } 
      } else if (mode == 5) {
          int point = 0;
          int ans = 0;
          int pos = 0;
          int ori_pos = 0;
          uLCD.cls();
          uLCD.printf("\nPlaying song #%d\n", gesture_index+1); //Default Green on black text
          if(gesture_index == 0) {
            for (int j = 0; j < 42; j++) {
              // uLCD.printf("\nPlaying song #%f\n", 500*song_note[j]); //Default Green on black text
              playNote(song_note[j]);
              uLCD.locate(1,2);
              ans = ((int)(500*song_note[j]) + point) % 2;
              uLCD.printf("hit: %2D", ans);
              uLCD.locate(1,8);
              uLCD.printf("points: %2D", point);
              t.start();
              wait_us(500000);
              error_reporter->Report("Time %f\n", t.read());
              t.reset();
              t.start();
              while(t.read() < 0.73){
                got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                            input_length, should_clear_buffer);

                // If there was no new data,
                // don't try to clear the buffer again and wait until next time
                if (!got_data) {
                  should_clear_buffer = false;
                  continue;
                }

                // Run inference, and report any error
                TfLiteStatus invoke_status = interpreter->Invoke();
                if (invoke_status != kTfLiteOk) {
                  error_reporter->Report("Invoke failed on index: %d\n", begin_index);
                  continue;
                }

                // Analyze the results to obtain a prediction
                gesture_index = PredictGesture(interpreter->output(0)->data.f);

                // Clear the buffer next time we read data
                should_clear_buffer = gesture_index < label_num;

                pos = (int)(t.read()*10);
                if(pos != ori_pos){
                  uLCD.locate(pos,5);
                  uLCD.printf("O");
                  ori_pos = pos;
                }

                if (gesture_index < label_num) {
                  if (gesture_index == ans){
                    error_reporter->Report("Taiko %d\n", gesture_index);
                    point++;
                    uLCD.locate(1,8);
                    uLCD.printf("points: %2D", point);
                  }
                }
              }
              t.reset();
              uLCD.cls();
            }
          } else {
            for (int j = 42; j < 66; j++) {
              // uLCD.printf("\nPlaying song #%f\n", 500*song_note[j]); //Default Green on black text
              playNote(song_note[j]);
              uLCD.locate(1,2);
              ans = ((int)(500*song_note[j]) + point) % 2;
              uLCD.printf("hit: %2D", ans);
              uLCD.locate(1,8);
              uLCD.printf("points: %2D", point);
              t.start();
              wait_us(500000);
              error_reporter->Report("Time %f\n", t.read());
              t.reset();
              t.start();
              while(t.read() < 0.73){
                got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                            input_length, should_clear_buffer);

                // If there was no new data,
                // don't try to clear the buffer again and wait until next time
                if (!got_data) {
                  should_clear_buffer = false;
                  continue;
                }

                // Run inference, and report any error
                TfLiteStatus invoke_status = interpreter->Invoke();
                if (invoke_status != kTfLiteOk) {
                  error_reporter->Report("Invoke failed on index: %d\n", begin_index);
                  continue;
                }

                // Analyze the results to obtain a prediction
                gesture_index = PredictGesture(interpreter->output(0)->data.f);

                // Clear the buffer next time we read data
                should_clear_buffer = gesture_index < label_num;

                pos = (int)(t.read()*10);
                if(pos != ori_pos){
                  uLCD.locate(pos,5);
                  uLCD.printf("O");
                  ori_pos = pos;
                }                

                if (gesture_index < label_num) {
                  if (gesture_index == ans){
                    error_reporter->Report("Taiko %d\n", gesture_index);
                    point++;
                    uLCD.locate(1,8);
                    uLCD.printf("points: %2D", point);
                  }
                }
              }
              t.reset();
              uLCD.cls();
            }
          }
    
      } else if (mode == 8) {
        for (int j = 0; j < 42; j++) {
            if(pause==0){
            uLCD.printf("\nPlaying song #%f\n", 500*song_note[j]); //Default Green on black text
            playNote(song_note[j]);
            wait(1.0);
          } else
          {
            uLCD.cls();
            uLCD.printf("\npause");
            wait(1.0);
          }
          
            button1.rise(&ISR2);
            button.rise(&ISR1);
          }
        for (int j = 42; j < 66; j++) {
            if(pause==0){
            uLCD.printf("\nPlaying song #%f\n", 500*song_note[j]); //Default Green on black text
            playNote(song_note[j]);
            wait(1.0);
            } else {
            uLCD.cls();
            uLCD.printf("\npause");
            wait(1.0);
            }
            button1.rise(&ISR2);
            button.rise(&ISR1);
          }
      }
    } 
  }
}