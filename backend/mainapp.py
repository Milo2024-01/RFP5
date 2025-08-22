from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
import numpy as np
import cv2
import io
from PIL import Image
import base64
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Hardcoded credentials
USERNAME = "admin"
PASSWORD = "mamerto"
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def authenticate_user(username: str, password: str) -> bool:
    return username == USERNAME and password == PASSWORD

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    if not authenticate_user(form_data.username, form_data.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": form_data.username, "token_type": "bearer"}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if token != USERNAME:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token

def is_valid_image_type(content_type: str, filename: str) -> bool:
    valid_types = ["image/jpeg", "image/jpg", "image/png"]
    valid_extensions = [".jpg", ".jpeg", ".png"]
    if content_type.lower() in valid_types:
        return True
    if any(filename.lower().endswith(ext) for ext in valid_extensions):
        return True
    return False

# Image processing using Excess Green (ExG)
def process_image(image_data: bytes, field_width_m: float, field_height_m: float):
    try:
        # Load image
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        img_np = np.array(image).astype(np.float32)

        # Validate green coverage
        hsv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.count_nonzero(green_mask) / (img_np.shape[0]*img_np.shape[1]) * 100
        if green_percentage < 30:
            return {"error": "Uploaded image does not appear to be a rice field."}

        # ExG calculation
        R = img_np[:, :, 0]
        G = img_np[:, :, 1]
        B = img_np[:, :, 2]
        exg = 2*G - R - B
        exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)

        # Threshold masks
        mask_healthy = exg_norm > 0.6
        mask_medium = (exg_norm <= 0.6) & (exg_norm > 0.3)
        mask_unhealthy = exg_norm <= 0.3

        # Color visualization
        out_img = np.zeros_like(img_np)
        out_img[mask_healthy] = [0, 255, 0]
        out_img[mask_medium] = [255, 255, 0]
        out_img[mask_unhealthy] = [255, 0, 0]

        # Area calculation
        total_pixels = img_np.shape[0]*img_np.shape[1]
        m2_per_pixel = (field_width_m*field_height_m)/total_pixels

        healthy_area = np.sum(mask_healthy)*m2_per_pixel
        medium_area = np.sum(mask_medium)*m2_per_pixel
        unhealthy_area = np.sum(mask_unhealthy)*m2_per_pixel

        # Yield estimation
        yield_kg = (healthy_area*0.8) + (medium_area*0.4) + (unhealthy_area*0.1)

        # Convert processed image to Base64
        output_pil = Image.fromarray(out_img.ast(np.uint8))
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG")
        processed_image_b64 = base64.b64encode(buffered.getvalue()).decode()

        return {
            "processed_image": processed_image_b64,
            "estimated_yield": round(yield_kg,2),
            "stats": {
                "healthy": round(healthy_area,2),
                "medium": round(medium_area,2),
                "unhealthy": round(unhealthy_area,2)
            }
        }

    except Exception as e:
        return {"error": f"Image processing failed: {str(e)}"}

@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    width: float = Form(...),
    height: float = Form(...),
    location: str = Form(None),
    current_user: str = Depends(get_current_user)
):
    if not is_valid_image_type(image.content_type, image.filename):
        return JSONResponse(status_code=400, content={"error": "Only JPG, JPEG, and PNG files are allowed"})
    contents = await image.read()
    result = process_image(contents, width, height)
    if "error" in result:
        return JSONResponse(status_code=400, content={"error": result["error"]})
    return JSONResponse(content=result)

# Health check endpoint
@app.get("/")
async def health_check():
    return {"status": "OK", "message": "Rice Field Health Analyzer API is running"}

# HEAD endpoint to prevent 405 logs
@app.head("/")
async def head_root():
    return {}

# Optional dummy favicon to prevent 404 logs
@app.get("/favicon.ico")
async def favicon():
    return ""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)