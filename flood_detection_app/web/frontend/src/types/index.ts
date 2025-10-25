// 类型定义文件

export interface VehicleResult {
  id: number;
  bbox: [number, number, number, number]; // [x1, y1, x2, y2]
  confidence: number;
  flood_level: 'light' | 'moderate' | 'severe';
  overlap_ratio: number;
}

export interface Statistics {
  total_vehicles: number;
  light_flood_count: number;
  moderate_flood_count: number;
  severe_flood_count: number;
  water_coverage_percentage: number;
  processing_time: number;
}

export interface AnalysisResult {
  success: boolean;
  message: string;
  vehicles: VehicleResult[];
  statistics: Statistics;
  processing_time: number;
  result_image_base64: string;
  water_coverage_percentage: number;
}

export interface AvailableModels {
  vehicle_models: string[];
  water_models: string[];
}

export interface ModelSelection {
  vehicle_model: string;
  water_model: string;
}

export interface HealthStatus {
  status: string;
  timestamp: number;
  models_loaded: boolean;
  version: string;
}