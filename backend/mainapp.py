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
import gc

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

def resize_image_if_needed(image: Image.Image, max_size: int = 1024) -> Image.Image:
    """Resize image if it's larger than max_size in any dimension"""
    width, height = image.size
    if width > max_size or height > max_size:
        # Calculate new dimensions while maintaining aspect ratio
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return image

# Memory-optimized image processing
def process_image(image_data: bytes, field_width_m: float, field_height_m: float):
    try:
        # Load image with memory optimization
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize if needed to reduce memory usage
        image = resize_image_if_needed(image, 1024)
        
        img_np = np.array(image)
        
        # Free memory as soon as possible
        del image
        gc.collect()
        
        # Convert to float32 for calculations (using less memory than float64)
        img_float = img_np.astype(np.float32) / 255.0

        # Validate green coverage
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
        lower_green = np.array([25, 40, 40])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        green_percentage = np.count_nonzero(green_mask) / (img_np.shape[0]*img_np.shape[1]) * 100
        
        # Free memory
        del hsv, green_mask
        gc.collect()
        
        if green_percentage < 30:
            return {"error": "Uploaded image does not appear to be a rice field."}

        # ExG calculation
        R = img_float[:, :, 0]
        G = img_float[:, :, 1]
        B = img_float[:, :, 2]
        exg = 2*G - R - B
        exg_norm = (exg - exg.min()) / (exg.max() - exg.min() + 1e-6)

        # Free memory
        del R, G, B, exg
        gc.collect()

        # Threshold masks - use uint8 to save memory
        mask_healthy = (exg_norm > 0.6).astype(np.uint8)
        mask_medium = ((exg_norm <= 0.6) & (exg_norm > 0.3)).astype(np.uint8)
        mask_unhealthy = (exg_norm <= 0.3).astype(np.uint8)

        # Free memory
        del exg_norm
        gc.collect()

        # Color visualization - create uint8 output image
        out_img = np.zeros_like(img_np)
        out_img[mask_healthy.astype(bool)] = [0, 255, 0]      # Green for healthy
        out_img[mask_medium.astype(bool)] = [255, 255, 0]     # Yellow for medium
        out_img[mask_unhealthy.astype(bool)] = [255, 0, 0]    # Red for unhealthy

        # Area calculation
        total_pixels = img_np.shape[0]*img_np.shape[1]
        m2_per_pixel = (field_width_m*field_height_m)/total_pixels

        healthy_area = np.sum(mask_healthy)*m2_per_pixel
        medium_area = np.sum(mask_medium)*m2_per_pixel
        unhealthy_area = np.sum(mask_unhealthy)*m2_per_pixel

        # Yield estimation
        yield_kg = (healthy_area*0.8) + (medium_area*0.4) + (unhealthy_area*0.1)

        # Convert processed image to Base64
        output_pil = Image.fromarray(out_img.astype(np.uint8))
        buffered = io.BytesIO()
        output_pil.save(buffered, format="PNG")
        processed_image_b64 = base64.b64encode(buffered.getvalue()).decode()

        # Free all memory before returning
        del img_np, img_float, mask_healthy, mask_medium, mask_unhealthy, out_img, output_pil, buffered
        gc.collect()

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
    
    # Limit file size to 5MB to prevent memory issues
    max_size = 5 * 1024 * 1024  # 5MB
    contents = await image.read()
    
    if len(contents) > max_size:
        return JSONResponse(
            status_code=400, 
            content={"error": "Image size too large. Please upload an image smaller than 5MB."}
        )
    
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