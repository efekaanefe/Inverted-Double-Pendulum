#include <Encoder.h>
#include <math.h> // For fmod

#define PI 3.1415926535897932384626433832795

// Motor Pin Definitions
const int ENA = 6;  // PWM pin for speed control
const int IN1 = 5;  // Motor direction pin 1
const int IN2 = 4;  // Motor direction pin 2

// Encoder Pin Definitions
Encoder linearEnc(2, 3);        // Linear encoder for motor (pins 2 and 3)
Encoder rotationalEnc(21, 20); // Rotational encoder (pins 20 and 21)

double motorSpeed = 0;  // PWM value for full speed
bool lqrActive = false; // Flag for LQR activation
bool swingUpActive = false; // Flag for Swing-up activation
bool systemActive = false; // Flag to start or stop the control system
bool sendKinematics = false; // Flag to enable or disable kinematics data transmission

// LQR Gain Matrix (example values, replace with actual gains)
double K[4] = {-185.16 * 3, -114.7 * 3, 321.13 * 3, 37.28}; // Gains for x, xdot, theta, thetadot

double targetX = 0.0;      // Desired x in meters
double targetTheta = PI;   // Desired theta in radians

// Conversion factors
const long maxTicks = 13500;      // Maximum encoder ticks for full range
const double maxDistance = 0.49;  // Maximum linear distance in meters
const double rotationalEncoderToRadians = 2 * PI / 1600; // Conversion factor for 2π = 1600 ticks

// Swing-up parameters
double swingUpTarget = 0.15 * maxDistance;
bool movingRight = true; 

// Define the LQR active range in degrees
double lqrRange = 60.0;
double lqrStartDeg = 180.0 - lqrRange ;  // Start of LQR range in degrees
double lqrEndDeg = 180.0 + lqrRange;   // End of LQR range in degrees
double lqrStartRad = lqrStartDeg * PI / 180.0;
double lqrEndRad = lqrEndDeg * PI / 180.0;
double xlim = 0.35 * maxDistance;


// Function to convert encoder ticks to meters
double encoderToMeters(long encoderTicks, long maxTicks, double maxDistanceM) {
  return (double(encoderTicks) / maxTicks) * maxDistanceM;
}

void setup() {
  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  Serial.begin(115200);
  Serial.println("Motor and Encoder Control with LQR and Swing-up");

  motorStop(); // Start with the motor stopped
}

void loop() {
  static unsigned long prevTime = 0;

  // Read serial commands
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();

    if (command == "start") {
      Serial.println("System started.");
      systemActive = true;
      swingUpActive = true;
    } else if (command == "stop") {
      Serial.println("System stopped.");
      systemActive = false;
      motorStop();
    } else if (command == "reset"){
        resetAll(K, targetX, targetTheta);
    } else if (command == "sendKinematics") {
      Serial.println("Sending kinematics data.");
      sendKinematics = true;
    } else if (command == "stopSendKinematics") {
      Serial.println("Stopped sending kinematics data.");
      sendKinematics = false;
    } else {
      // Check for changeTargetX command
      if (command.startsWith("changeTargetX")) {
        int spaceIndex = command.indexOf(' '); // Find the space separating command and value
        if (spaceIndex != -1) {
          String valueStr = command.substring(spaceIndex + 1); // Extract the value part
          int targetXNew = valueStr.toInt(); // Convert to integer
          if (targetXNew >= -100 && targetXNew <= 100) {
            // Update your target X variable here
            targetX = targetXNew * xlim / 100;

            Serial.print("Target X changed to: ");
            Serial.println(targetX);

          } else {
            Serial.println("Error: Value out of range. Please give a value between [-100, 100].");
          }
        } else {
          Serial.println("Error: No value provided for changeTargetX.");
          }
        } 
      // Check for changeGains command
      else if (command.startsWith("changeGains")) {
      int firstSpace = command.indexOf(' ');
      int secondSpace = command.indexOf(' ', firstSpace + 1); // Find the second space
      if (firstSpace != -1 && secondSpace != -1) {
        String indexStr = command.substring(firstSpace + 1, secondSpace); // Extract the index
        String gainStr = command.substring(secondSpace + 1); // Extract the gain
        float gain = gainStr.toFloat(); // Convert gain to float

        // Map index to integer
        int index = -1;
        if (indexStr == "x") index = 0;
        else if (indexStr == "xdot") index = 1;
        else if (indexStr == "theta") index = 2;
        else if (indexStr == "thetadot") index = 3;

        if (index != -1) {
          K[index] = gain; // Update the gain in the K array
          printLQRGains(K); // Call the function to print updated gains
        } else {
          Serial.println("Error: Invalid index. Use x, xdot, theta, or thetadot.");
        }
      } else {
        Serial.println("Error: Invalid format for changeGains. Use: changeGains <index> <gain>");
      }
    }
      else {
        Serial.println("Unknown command.");
        }
      }
  }

  unsigned long currentTime = millis();
  double dt = (currentTime - prevTime) / 1000.0;

  // Read encoder values
  long linearTicks = linearEnc.read();
  long rotationalTicks = rotationalEnc.read();

  // Calculate theta (in radians) and x (in meters)
  double theta = rotationalTicks * rotationalEncoderToRadians; // Convert to radians
  theta = fmod(theta, 2 * PI); // Wrap theta between 0 and 2π
  if (theta < 0) theta += 2 * PI; // Ensure positive value

  double x = encoderToMeters(linearTicks, maxTicks, maxDistance); // Convert to meters

  // Calculate derivatives (xdot and thetadot)
  static double prevX = 0.0, prevTheta = 0.0;
  double xdot = (x - prevX) / dt;
  double thetadot = (theta - prevTheta) / dt;
  prevX = x;
  prevTheta = theta;

  if (sendKinematics) {
  // Send the 4 values over serial, separated by commas
    Serial.print(x); // 2 decimal places
    Serial.print(",");
    Serial.print(xdot);
    Serial.print(",");
    Serial.print(theta);
    Serial.print(",");
    Serial.println(thetadot); // println to end the line

  }

  if (!systemActive) {
    return; // Skip control logic if the system is stopped
  }

  // LQR or Swing-up control logic
  if (theta >= lqrStartRad && theta <= lqrEndRad && x >= -xlim && x <= xlim) {
    lqrControl(x, xdot, theta, thetadot, targetX);
  } else {
    swingUpControl(x);
  }

  prevTime = currentTime;
  delay(5); // Small delay for stability
}

void swingUpControl(double x) {
  swingUpActive = true;
  lqrActive = false;

  const double targetRight = swingUpTarget;
  const double targetLeft = -swingUpTarget;
  const int fullPower = 110;

  if (movingRight && x >= targetRight) {
    movingRight = false;
    // Serial.println("Reached targetRight, switching to left");
  } else if (!movingRight && x <= targetLeft) {
    movingRight = true;
    // Serial.println("Reached targetLeft, switching to right");
  }

  if (movingRight) {
    motorSpeed = fullPower;
    motorForward();
  } else {
    motorSpeed = fullPower;
    motorReverse();
  }
}

void lqrControl(double x, double xdot, double theta, double thetadot, double targetX_given) {
  swingUpActive = false;
  lqrActive = true;

  double state[4] = {x, xdot, theta, thetadot};
  double target[4] = {0, targetX_given, PI, 0};

  double controlInput = 0.0;
  for (int i = 0; i < 4; i++) {
    controlInput -= K[i] * (state[i] - target[i]);
  }

  controlInput = controlInput * 3;

  int pwm_max = 180;
  motorSpeed = constrain(abs(controlInput), 0, pwm_max);
  motorSpeed = map(motorSpeed, 0, pwm_max, 60, pwm_max);

  if (controlInput > 0) {
    motorForward();
  } else if (controlInput < 0) {
    motorReverse();
  } 
  // else {
  //   motorStop();
  // }

  // Serial.println("LQR Control Active");
}

void motorForward() {
  digitalWrite(IN1, HIGH);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, motorSpeed);
}

void motorReverse() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, HIGH);
  analogWrite(ENA, motorSpeed);
}

void motorStop() {
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
}

// Function to print the LQR gains
void printLQRGains(double K[]) {
  Serial.println("LQR Gains:");
  for (int i = 0; i < 4; i++) {
    Serial.print("K[");
    Serial.print(i);
    Serial.print("] = ");
    Serial.println(K[i], 2); // Print with 2 decimal places
  }
}


// Function to reset all values to defaults
void resetAll(double K[], double &targetX, double &targetTheta) {

  // Default values for gains and targets
  const double DEFAULT_K[4] = {-185.16 * 3, -114.7 * 3, 321.13 * 3, 37.28};
  const double DEFAULT_TARGET_X = 0.0;
  const double DEFAULT_TARGET_THETA = PI;

  // Reset gains
  for (int i = 0; i < 4; i++) {
    K[i] = DEFAULT_K[i];
  }

  // Reset targets
  targetX = DEFAULT_TARGET_X;
  targetTheta = DEFAULT_TARGET_THETA;

  Serial.println("All values reset to defaults:");
  printLQRGains(K);
  Serial.print("Target X: ");
  Serial.println(targetX, 2);
  Serial.print("Target Theta: ");
  Serial.println(targetTheta, 2);
}
