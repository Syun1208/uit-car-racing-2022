#include <stdio.h>
#include <math.h>
#include <webots/robot.h>
#include <webots/motor.h>
#include <webots/distance_sensor.h>
#include <webots/led.h>
#include <webots/camera.h>
#include <webots/supervisor.h>
#define TIME_STEP 8

// 8 IR ground color sensors
#define NB_GROUND_SENS 8
#define NB_LEDS 5

// IR Ground Sensors
WbDeviceTag gs[NB_GROUND_SENS];

// LEDs 
WbDeviceTag led[NB_LEDS];

// Motors
WbDeviceTag left_motor, right_motor;

/* Phần code được phép chỉnh sửa cho phù hợp */
// Định nghĩa các tín hiệu của xe
#define NOP  -1
#define MID   0
#define LEFT  1
#define RIGHT 2
#define FULL_SIGNAL 3
#define BLANK_SIGNAL 4
#define STOP_SIGNAL 5

// Điểu chỉnh tốc độ phù hợp
// MAX_SPEED <= 1000
#define MAX_SPEED 6

// Khai báo biến cho các sensors
unsigned short threshold[NB_GROUND_SENS] = { 300 , 300 , 300 , 300 , 300 , 300 , 300 , 300 };
unsigned int filted[8] = {0 , 0 , 0 , 0 , 0 , 0 , 0 , 0};

// Biến lưu giá trị tỉ lệ tốc độ của động cơ
double left_ratio = 0.0;
double right_ratio = 0.0;



//min = 0 , max = 10
void constrain(double *value, double min, double max) {
  if (*value > max) *value = max;
  if (*value < min) *value = min;
}

//Thí sinh không được bỏ phần đọc tín hiệu này
//Hàm đọc giá trị sensors
void ReadSensors(){
  unsigned short gs_value[NB_GROUND_SENS] = {0, 0, 0, 0, 0, 0, 0, 0};
  for(int i=0; i<NB_GROUND_SENS; i++){
    gs_value[i] = wb_distance_sensor_get_value(gs[i]);
    // So sánh giá trị gs_value với threshold -> chuyển đổi sang nhị phân
    if (gs_value[i] < threshold[i])
      filted[i] = 1;
    else filted[i] = 0;
  }

}

int Position ()
{
    if ( filted[3] == 1 && filted[4] == 1)
     { return MID;}
}

//hàm điều khiểu xe đi thẳng
void GoStraight() {
  left_ratio = 2.0;
  right_ratio = 2.0;
}

//hàm dừng xe
void Stop()
{
  left_ratio =  0;
  right_ratio = 0;
}

/*
 * This is the main program.
 */
int main() {
  
  //dùng để khai báo robot 
  //#không được bỏ
  wb_robot_init();    

  /* get and enable the camera and accelerometer */
  WbDeviceTag camera = wb_robot_get_device("camera");
  wb_camera_enable(camera, 64);

  /* initialization */
  char name[20];
  for (int i = 0; i < NB_GROUND_SENS; i++) {
    sprintf(name, "gs%d", i);
    gs[i] = wb_robot_get_device(name); /* ground sensors */
    wb_distance_sensor_enable(gs[i], TIME_STEP);
  }

  for (int i = 0; i < NB_LEDS; i++) {
    sprintf(name, "led%d", i);
    led[i] = wb_robot_get_device(name);
    wb_led_set(led[i], 1);
  }  
  // motors
  left_motor = wb_robot_get_device("left wheel motor");
  right_motor = wb_robot_get_device("right wheel motor");
  wb_motor_set_position(left_motor, INFINITY);
  wb_motor_set_position(right_motor, INFINITY);
  wb_motor_set_velocity(left_motor, 0.0);
  wb_motor_set_velocity(right_motor, 0.0);
    
  // Chương trình sẽ được lặp lại vô tận trong hàm while
  while (wb_robot_step(TIME_STEP) != -1)
  {
    ReadSensors();   
    if (wb_robot_get_time() < 0.5 ) 
       GoStraight();  
    
    //In giá trị của cảm biến ra màn hình
    printf ("\n\t\tPosition : 0b");
    for (int i = 0 ; i < 8 ; i ++)
    {
      printf ("%u" , filted[i] );
    }
    // Điều khiển xe
   
    //đi thẳng
    if (Position() == MID)
      GoStraight();
    else 
      Stop();
    // Điều chỉnh tốc độ động cơ
    wb_motor_set_velocity(left_motor, left_ratio * MAX_SPEED);
    wb_motor_set_velocity(right_motor, right_ratio * MAX_SPEED);
  }
  wb_robot_cleanup();
  return 0;
}
    