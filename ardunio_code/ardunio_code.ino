#define motorPin1   2
#define motorPin2   4
#define motorSpeed  3

#define limitFullOpen1 8
#define limitHalfOpen2    9
#define limitClose3       10
#define gateCloseAfterSec 5
String str = "";

void motorCW()
{
  digitalWrite(motorPin1, HIGH);
  digitalWrite(motorPin2, LOW);
}

void motorCCW()
{
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, HIGH);
}
void motorStop()
{
  digitalWrite(motorPin1, LOW);
  digitalWrite(motorPin2, LOW);
}


void setup()
{
  Serial.begin(9600);
  
  pinMode(motorPin1, OUTPUT);
  pinMode(motorPin2, OUTPUT);
  
  pinMode(limitFullOpen1, INPUT_PULLUP);
  pinMode(limitHalfOpen2, INPUT_PULLUP);
  pinMode(limitClose3, INPUT_PULLUP);
  analogWrite(motorSpeed, 250);
}

void loop()
{
  
   if (Serial.available()) 
   {
    String str = Serial.readStringUntil('\n');
      if(str == "ho")
    {
      motorCW();
      while(digitalRead(limitHalfOpen2));
      motorStop();
      delay(gateCloseAfterSec*1000);
      motorCCW();
      while(digitalRead(limitClose3));
      motorStop();
    }
    else if(str == "fo")
    {
      motorCW();
      while(digitalRead(limitFullOpen1));
      motorStop();
      delay(gateCloseAfterSec*1000);
      motorCCW();
      while(digitalRead(limitClose3));
      motorStop();
    }
   
  }
  
}
