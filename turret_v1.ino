#include <Servo.h>

#define ENA 6
#define IN1 4
#define IN2 5

Servo pelatuk;

void setup() {
  pelatuk.attach(8);
  pelatuk.write(0); // Awal posisi servo

  Serial.begin(9600);

  pinMode(ENA, OUTPUT);
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);

  // Pastikan motor mati dulu
  digitalWrite(IN1, LOW);
  digitalWrite(IN2, LOW);
  analogWrite(ENA, 0);
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();

    if (command == '1') {
      // 1. Nyalakan motor
      digitalWrite(IN1, HIGH);
      digitalWrite(IN2, LOW);
      analogWrite(ENA, 255);  // Sesuaikan kecepatan

      delay(10000); // Tunggu 5 detik

      // 2. Servo gerak 6x
      for (int i = 0; i < 6; i++) {
        pelatuk.write(0);   // Tembak
        delay(350);
        pelatuk.write(90);  // Kembali
        delay(350);
      }

      // 3. Matikan motor
      digitalWrite(IN1, LOW);
      digitalWrite(IN2, LOW);
      analogWrite(ENA, 0);
      pelatuk.write(0);
    }
  }
}
