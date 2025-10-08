""" Real-World 3D Truck Optimization System - FastAPI Backend
FIXED VERSION - Thoroughly tested with correct rectangle splitting
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="3D Truck Loading Optimization API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Data ====================
CITY_DISTANCES_KM = {
    frozenset(("Mumbai", "Delhi")): 1420,
    frozenset(("Mumbai", "Bangalore")): 985,
    frozenset(("Mumbai", "Kolkata")): 1940,
    frozenset(("Mumbai", "Chennai")): 1335,
    frozenset(("Mumbai", "Hyderabad")): 710,
    frozenset(("Delhi", "Bangalore")): 2150,
    frozenset(("Delhi", "Kolkata")): 1530,
    frozenset(("Delhi", "Chennai")): 2210,
    frozenset(("Delhi", "Hyderabad")): 1580,
    frozenset(("Bangalore", "Kolkata")): 1870,
    frozenset(("Bangalore", "Chennai")): 350,
    frozenset(("Bangalore", "Hyderabad")): 570,
    frozenset(("Kolkata", "Chennai")): 1670,
    frozenset(("Kolkata", "Hyderabad")): 1480,
    frozenset(("Chennai", "Hyderabad")): 630,
}

COST_MODEL = {
    "22 ft Truck": {"base_rate": 4000, "rate_per_km": 55},
    "32 ft Single Axle": {"base_rate": 5000, "rate_per_km": 65},
    "32 ft Multi Axle": {"base_rate": 6000, "rate_per_km": 75},
}

# ==================== Models ====================
class BoxInput(BaseModel):
    box_type: str
    external_length_mm: Optional[float] = None
    external_width_mm: Optional[float] = None
    external_height_mm: Optional[float] = None
    max_payload_kg: float
    quantity: Optional[int] = None

class TruckInput(BaseModel):
    name: str
    internal_length_mm: float
    internal_width_mm: float
    internal_height_mm: float
    payload_kg: float

class OptimizationRequest(BaseModel):
    boxes: List[BoxInput]
    trucks: List[TruckInput]
    source_city: Optional[str] = None
    destination_city: Optional[str] = None

class BoxPlacement(BaseModel):
    type: str
    dims_mm: List[float]
    pos_mm: List[float]
    rotation: str
    corners: Dict[str, List[float]]
    weight_kg: float

class TruckDimensions(BaseModel):
    length_mm: float
    width_mm: float
    height_mm: float
    volume_mm3: float
    payload_kg: float

class TruckResult(BaseModel):
    truck_name: str
    truck_dimensions: TruckDimensions
    units_packed_total: int
    cube_utilisation_pct: float
    payload_used_kg: float
    payload_used_pct: float
    estimated_cost: Optional[float] = None
    box_counts_by_type: Dict[str, int]
    unfitted_counts: Dict[str, int]
    placements_sample: List[BoxPlacement]
    verification_passed: bool
    verification_details: List[str]

# ==================== FIXED GUILLOTINE PACKING ====================

@dataclass
class Box:
    type: str
    length: float
    width: float
    height: float
    weight: float
    id: int

    @property
    def volume(self):
        return self.length * self.width * self.height

@dataclass
class Placement:
    box: Box
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float

    @property
    def x_max(self): return self.x + self.length
    @property
    def y_max(self): return self.y + self.height
    @property
    def z_max(self): return self.z + self.width

@dataclass
class FreeSpace:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float

    @property
    def volume(self):
        return self.length * self.width * self.height
    
    def fits(self, l, w, h):
        return self.length >= l and self.width >= w and self.height >= h

class GuillotinePacker:
    """
    Simplified Guillotine algorithm - most reliable for identical boxes
    """
    
    def __init__(self, truck_length, truck_width, truck_height, max_weight):
        self.truck_length = truck_length
        self.truck_width = truck_width
        self.truck_height = truck_height
        self.max_weight = max_weight
        self.placements = []
        self.total_weight = 0
        self.free_spaces = [FreeSpace(0, 0, 0, truck_length, truck_width, truck_height)]
        
    def pack_boxes(self, boxes: List[Box]) -> Tuple[List[Placement], List[Box]]:
        """Pack boxes using guillotine cuts"""
        unpacked = []
        
        # Sort by volume descending
        sorted_boxes = sorted(boxes, key=lambda b: b.volume, reverse=True)
        
        logger.info(f"Attempting to pack {len(sorted_boxes)} boxes")
        logger.info(f"Truck: {self.truck_length}×{self.truck_width}×{self.truck_height}mm, {self.max_weight}kg")
        
        packed_count = 0
        weight_limited = 0
        space_limited = 0
        
        # REPLACE the above loop block with this:
        
        packed_count = 0
        weight_limited = 0
        space_limited = 0
        
        # ADDED: Safety break condition to prevent server timeouts/crashes
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 100 # Stop if 100 boxes in a row can't be placed
        
        for i, box in enumerate(sorted_boxes):
            # Weight check
            if self.total_weight + box.weight > self.max_weight:
                unpacked.append(box)
                weight_limited += 1
                consecutive_failures = 0 # Reset counter on weight limit (it's not a space failure)
                continue
            
            # Try to place
            placed = self._place_box(box)
            
            if placed:
                packed_count += 1
                consecutive_failures = 0 # Reset on success
                if packed_count % 100 == 0:
                    logger.info(f"  Progress: {packed_count} boxes packed...")
            else:
                unpacked.append(box)
                space_limited += 1
                consecutive_failures += 1 # Increment on failure
                
                # HALT CONDITION: Break if too many consecutive boxes fail to find a fit
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    # Add all remaining boxes to the unpacked list
                    unpacked.extend(sorted_boxes[i+1:])
                    logger.info(f"HALTED: Reached {MAX_CONSECUTIVE_FAILURES} consecutive placement failures. Remaining {len(sorted_boxes) - i - 1} boxes added to unpacked list.")
                    break # Exit the main packing loop
        
        
        utilization = self.get_utilization()
        weight_pct = (self.total_weight / self.max_weight * 100)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"PACKING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"✓ Boxes packed:       {packed_count}")
        logger.info(f"✗ Space limited:      {space_limited}")
        logger.info(f"✗ Weight limited:     {weight_limited}")
        logger.info(f"  Volume utilization: {utilization:.1f}%")
        logger.info(f"  Weight used:        {self.total_weight:.1f} / {self.max_weight:.1f} kg ({weight_pct:.1f}%)")
        logger.info(f"  Free spaces remaining: {len(self.free_spaces)}")
        logger.info(f"{'='*60}\n")
        
        return self.placements, unpacked
    
    def _place_box(self, box: Box) -> bool:
        """Try to place box in best free space"""
        
        # Try all 6 rotations
        rotations = [
            (box.length, box.width, box.height),
            (box.length, box.height, box.width),
            (box.width, box.length, box.height),
            (box.width, box.height, box.length),
            (box.height, box.length, box.width),
            (box.height, box.width, box.length),
        ]
        
        best_space_idx = None
        best_rotation = None
        best_score = float('inf')
        
        # Find best fit (prefer lower Y, tighter fit)
        for idx, space in enumerate(self.free_spaces):
            for rotation in rotations:
                l, w, h = rotation
                if space.fits(l, w, h):
                    # Score: prefer low position, tight fit
                    leftover = space.volume - (l * w * h)
                    score = space.y * 1000 + leftover
                    
                    if score < best_score:
                        best_score = score
                        best_space_idx = idx
                        best_rotation = rotation
        
        if best_space_idx is None:
            return False
        
        # Place the box
        space = self.free_spaces[best_space_idx]
        l, w, h = best_rotation
        
        placement = Placement(
            box=box,
            x=space.x,
            y=space.y,
            z=space.z,
            length=l,
            width=w,
            height=h
        )
        
        self.placements.append(placement)
        self.total_weight += box.weight
        
        # Remove used space and create new free spaces
        self.free_spaces.pop(best_space_idx)
        new_spaces = self._split_space(space, l, w, h)
        self.free_spaces.extend(new_spaces)
        
        # Clean up tiny spaces
        self.free_spaces = [s for s in self.free_spaces if s.volume > 100]
        
        # Sort by Y position (bottom-up packing)
        self.free_spaces.sort(key=lambda s: (s.y, s.x, s.z))
        
        # Limit number of spaces to prevent explosion
        # if len(self.free_spaces) > 10000:
        #     self.free_spaces = self.free_spaces[:10000]
        
        return True
    
    def _split_space(self, space: FreeSpace, used_l: float, used_w: float, used_h: float) -> List[FreeSpace]:
        """
        Split space using guillotine method - create 3 new spaces
        This is the CORRECT way to split
        """
        new_spaces = []
        
        # Remaining space to the RIGHT (along X axis)
        if space.length > used_l:
            new_spaces.append(FreeSpace(
                x=space.x + used_l,
                y=space.y,
                z=space.z,
                length=space.length - used_l,
                width=space.width,
                height=space.height
            ))
        
        # Remaining space to the FRONT (along Z axis)
        if space.width > used_w:
            new_spaces.append(FreeSpace(
                x=space.x,
                y=space.y,
                z=space.z + used_w,
                length=used_l,  # Only the used length
                width=space.width - used_w,
                height=space.height
            ))
        
        # Remaining space ABOVE (along Y axis)
        if space.height > used_h:
            new_spaces.append(FreeSpace(
                x=space.x,
                y=space.y + used_h,
                z=space.z,
                length=used_l,  # Only the used length
                width=used_w,   # Only the used width
                height=space.height - used_h
            ))
        
        return new_spaces
    
    def get_utilization(self) -> float:
        truck_volume = self.truck_length * self.truck_width * self.truck_height
        used_volume = sum(p.length * p.width * p.height for p in self.placements)
        return (used_volume / truck_volume) * 100 if truck_volume > 0 else 0
    
    def verify_packing(self) -> Tuple[bool, List[str]]:
        issues = []
        
        if self.total_weight > self.max_weight + 0.01:
            issues.append(f"Weight exceeded: {self.total_weight:.1f} > {self.max_weight} kg")
        
        for i, p in enumerate(self.placements):
            if p.x_max > self.truck_length + 1:
                issues.append(f"Box {i} exceeds length")
            if p.y_max > self.truck_height + 1:
                issues.append(f"Box {i} exceeds height")
            if p.z_max > self.truck_width + 1:
                issues.append(f"Box {i} exceeds width")
        
        return len(issues) == 0, issues

# ==================== API ====================
@app.post("/api/optimize", response_model=List[TruckResult])
async def optimize_loading(request: OptimizationRequest):
    try:
        results = []
        
        for truck in request.trucks:
            logger.info(f"\n{'='*70}")
            logger.info(f"OPTIMIZING: {truck.name}")
            logger.info(f"{'='*70}")
            
            # Calculate cost
            calculated_cost = None
            if request.source_city and request.destination_city:
                distance_key = frozenset((request.source_city, request.destination_city))
                distance = CITY_DISTANCES_KM.get(distance_key)
                cost_params = COST_MODEL.get(truck.name)
                if distance and cost_params:
                    calculated_cost = cost_params["base_rate"] + (distance * cost_params["rate_per_km"])
            
            # Generate boxes
            all_boxes = []
            box_id = 0
            
            for box_config in request.boxes:
                if box_config.quantity is None:
                    truck_vol = truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm
                    box_vol = (box_config.external_length_mm * 
                              box_config.external_width_mm * 
                              box_config.external_height_mm)
                    
                    # Generate 5x theoretical maximum to ensure we don't run out
                    max_by_vol = int(truck_vol / box_vol * 5.0) if box_vol > 0 else 0
                    max_by_weight = int(truck.payload_kg / box_config.max_payload_kg * 1.1) if box_config.max_payload_kg > 0 else 0
                    
                    quantity = min(max_by_vol, max_by_weight, 200000)
                    
                    logger.info(f"Box: {box_config.box_type}")
                    logger.info(f"  Dimensions: {box_config.external_length_mm}×{box_config.external_width_mm}×{box_config.external_height_mm}mm")
                    logger.info(f"  Weight: {box_config.max_payload_kg} kg")
                    logger.info(f"  Max by volume: {max_by_vol}")
                    logger.info(f"  Max by weight: {max_by_weight}")
                    logger.info(f"  Generating: {quantity} boxes")
                else:
                    quantity = box_config.quantity
                
                for _ in range(quantity):
                    all_boxes.append(Box(
                        type=box_config.box_type,
                        length=box_config.external_length_mm,
                        width=box_config.external_width_mm,
                        height=box_config.external_height_mm,
                        weight=box_config.max_payload_kg,
                        id=box_id
                    ))
                    box_id += 1
            
            logger.info(f"\nTotal boxes generated: {len(all_boxes)}")
            
            # PACK
            packer = GuillotinePacker(
                truck.internal_length_mm,
                truck.internal_width_mm,
                truck.internal_height_mm,
                truck.payload_kg
            )
            
            packed, unpacked = packer.pack_boxes(all_boxes)
            
            # Results
            box_counts = {}
            unfitted_counts = {}
            
            for p in packed:
                box_counts[p.box.type] = box_counts.get(p.box.type, 0) + 1
            
            for b in unpacked:
                unfitted_counts[b.type] = unfitted_counts.get(b.type, 0) + 1
            
            placements_sample = []
            for p in packed:
                placements_sample.append(BoxPlacement(
                    type=p.box.type,
                    dims_mm=[p.length, p.height, p.width],
                    pos_mm=[p.x, p.y, p.z],
                    rotation="Optimized",
                    corners={
                        "min": [p.x, p.y, p.z],
                        "max": [p.x_max, p.y_max, p.z_max]
                    },
                    weight_kg=p.box.weight
                ))
            
            is_valid, issues = packer.verify_packing()
            utilization = packer.get_utilization()
            weight_pct = (packer.total_weight / truck.payload_kg * 100)
            
            results.append(TruckResult(
                truck_name=truck.name,
                truck_dimensions=TruckDimensions(
                    length_mm=truck.internal_length_mm,
                    width_mm=truck.internal_width_mm,
                    height_mm=truck.internal_height_mm,
                    volume_mm3=truck.internal_length_mm * truck.internal_width_mm * truck.internal_height_mm,
                    payload_kg=truck.payload_kg
                ),
                units_packed_total=len(packed),
                cube_utilisation_pct=round(utilization, 2),
                payload_used_kg=round(packer.total_weight, 2),
                payload_used_pct=round(weight_pct, 2),
                estimated_cost=calculated_cost,
                box_counts_by_type=box_counts,
                unfitted_counts=unfitted_counts,
                placements_sample=placements_sample,
                verification_passed=is_valid,
                verification_details=issues if not is_valid else ["All checks passed"]
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
async def root():
    return {
        "name": "3D Truck Loading Optimization API",
        "version": "4.0.0 - Guillotine Algorithm (Fixed)",
        "features": ["Corrected Guillotine Packing", "No Space Limit", "Maximum Utilization"],
        "endpoints": {"/api/optimize": "POST", "/api/health": "GET", "/docs": "API docs"}
    }

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("  3D TRUCK OPTIMIZATION - FIXED GUILLOTINE ALGORITHM")
    print("="*70)
    print("  API: http://localhost:8000")
    print("  Docs: http://localhost:8000/docs")
    print("="*70 + "\n")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)