#define UV_PIN   A0
#define LED_PIN  8

int uvValue;
int uvIndex;

void setup()
{
  pinMode(LED_PIN, OUTPUT);
  Serial.begin(9600);
}

void loop()
{
  uvValue = analogRead(UV_PIN);

  // ---- UV Index Mapping (Based on Experimental Data) ----
  if (uvValue >= 26)          uvIndex = 4;   // Very High (No filter)
  else if (uvValue >= 11)     uvIndex = 3;   // Medium (Blue)
  else if (uvValue >= 6)      uvIndex = 2;   // Low (Green)
  else if (uvValue >= 1)      uvIndex = 1;   // Very Low (Red)
  else                        uvIndex = 0;   // No UV

  // ---- LED Indication ----
  if (uvIndex >= 3)
    digitalWrite(LED_PIN, HIGH);
  else
    digitalWrite(LED_PIN, LOW);

  // ---- Serial Monitor Output ----
  Serial.print("Analog Value: ");
  Serial.print(uvValue);
  Serial.print(" | UV Index (Relative): ");
  Serial.println(uvIndex);

  delay(500);
}
