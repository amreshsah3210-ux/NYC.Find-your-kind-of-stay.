from contextlib import asynccontextmanager
from pathlib import Path
import logging

import joblib
import kagglehub
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "Model_Pipeline.pkl"
STATIC_DIR = BASE_DIR / "static"
DATASET_NAME = "dgomonov/new-york-city-airbnb-open-data"
DATASET_FILE = "AB_NYC_2019.csv"

FEATURE_COLUMNS = [
	"neighbourhood_group",
	"neighbourhood",
	"latitude",
	"longitude",
	"price",
	"minimum_nights",
	"number_of_reviews",
	"reviews_per_month",
	"calculated_host_listings_count",
	"availability_365",
]


class ListingFeatures(BaseModel):
	neighbourhood_group: str = Field(min_length=1)
	neighbourhood: str = Field(min_length=1)
	latitude: float = Field(ge=-90, le=90)
	longitude: float = Field(ge=-180, le=180)
	price: float = Field(gt=0)
	minimum_nights: int = Field(ge=1)
	number_of_reviews: int = Field(ge=0)
	reviews_per_month: float = Field(ge=0)
	calculated_host_listings_count: int = Field(ge=1)
	availability_365: int = Field(ge=0, le=365)


def load_dataset() -> pd.DataFrame:
	download_path = Path(kagglehub.dataset_download(DATASET_NAME))
	csv_path = download_path / DATASET_FILE
	if not csv_path.is_file():
		matches = list(download_path.rglob(DATASET_FILE))
		if not matches:
			raise FileNotFoundError(f"Could not find {DATASET_FILE} in {download_path}")
		csv_path = matches[0]
	logger.info("Reading Kaggle dataset from %s", csv_path)
	return pd.read_csv(csv_path)


@asynccontextmanager
async def lifespan(application: FastAPI):
	application.state.pipeline = joblib.load(MODEL_PATH)
	application.state.dataset = None
	application.state.dataset_error = None
	try:
		application.state.dataset = load_dataset()
	except Exception as error:
		application.state.dataset_error = str(error)
		logger.exception("Kaggle dataset could not be loaded")
	yield


app = FastAPI(
	title="NYC Stay Type Classifier",
	description="Explore New York City Airbnb listings and estimate their room type.",
	version="1.0.0",
	lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
	return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str | bool]:
	return {
		"status": "ok",
		"model_loaded": hasattr(app.state, "pipeline"),
		"dataset_loaded": app.state.dataset is not None,
	}


@app.get("/api/summary")
def summary() -> dict:
	dataset = app.state.dataset
	if dataset is None:
		return {
			"available": False,
			"message": "The classifier is ready, but the Kaggle dataset is unavailable.",
			"detail": app.state.dataset_error,
			"listing_count": 0,
			"room_types": [],
			"boroughs": [],
			"neighbourhoods": [],
			"average_price": None,
			"median_price": None,
			"listings": [],
		}

	room_counts = dataset["room_type"].value_counts()
	borough_counts = dataset["neighbourhood_group"].value_counts()
	price_series = pd.to_numeric(dataset["price"], errors="coerce").dropna()
	listings = dataset[
		["name", "neighbourhood", "neighbourhood_group", "room_type", "price", "minimum_nights"]
	].head(8).fillna("").to_dict(orient="records")

	return {
		"available": True,
		"listing_count": int(len(dataset)),
		"room_types": [
			{"name": str(name), "count": int(count), "share": round(float(count / len(dataset) * 100), 1)}
			for name, count in room_counts.items()
		],
		"boroughs": [
			{"name": str(name), "count": int(count)} for name, count in borough_counts.items()
		],
		"neighbourhoods": sorted(dataset["neighbourhood"].dropna().astype(str).unique().tolist()),
		"average_price": round(float(price_series.mean()), 2) if not price_series.empty else None,
		"median_price": round(float(price_series.median()), 2) if not price_series.empty else None,
		"listings": listings,
	}


@app.post("/api/predict")
def predict(features: ListingFeatures) -> dict:
	pipeline = getattr(app.state, "pipeline", None)
	if pipeline is None:
		raise HTTPException(status_code=503, detail="The classification model is not available.")

	input_data = pd.DataFrame([features.model_dump()], columns=FEATURE_COLUMNS)
	prediction = str(pipeline.predict(input_data)[0])
	probabilities = {}
	if hasattr(pipeline, "predict_proba"):
		raw_probabilities = pipeline.predict_proba(input_data)[0]
		probabilities = {
			str(label): round(float(probability) * 100, 1)
			for label, probability in zip(pipeline.classes_, raw_probabilities)
		}

	return {
		"room_type": prediction,
		"confidence": probabilities.get(prediction),
		"probabilities": probabilities,
	}


@app.get("/api/listings")
def listings(limit: int = Query(default=8, ge=1, le=50)) -> dict:
	dataset = app.state.dataset
	if dataset is None:
		raise HTTPException(status_code=503, detail="The Kaggle dataset is unavailable.")
	columns = ["name", "neighbourhood", "neighbourhood_group", "room_type", "price", "minimum_nights"]
	return {"listings": dataset[columns].head(limit).fillna("").to_dict(orient="records")}
