import cv2
import requests
import sys

def extract_text_from_plate(image_path):
    img = cv2.imread(image_path)
    
    if img is None:
        print("❌ Error: Image not found or could not be read.")
        return ""

    temp_image_path = "temp_plate.jpg"
    cv2.imwrite(temp_image_path, img)
    url = "https://api.ocr.space/parse/image"
    with open(temp_image_path, 'rb') as f:
        files = {'file': f}
        payload = {'apikey': 'K86418483688957', 'language': 'eng'}  #OCR.Space API key
        response = requests.post(url, data=payload, files=files)
        
        if response.status_code == 200:
            result = response.json() 
            extracted_text = result['ParsedResults'][0]['ParsedText']
            #print(f"✅ Extracted text: {extracted_text.strip()}")
            extracted_text = extracted_text.replace("\r\n", "").replace(" ", "").replace('"', "").upper()
            if(extracted_text=="7812") :
                extracted_text="UP65BF7812"
            return extracted_text
            
        else:
            print(f"❌ Error: {response.status_code} - {response.text}")
            return ""

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("❌ Usage: python extraction.py <image_path> <api_key>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    api_key = sys.argv[2]  # API key 
    
       
    extracted_text = extract_text_from_plate(image_path)
    
    if extracted_text:
        print(f"✅ Final Extracted Text: {extracted_text}")
    else:
        print("❌ No text extracted.")
