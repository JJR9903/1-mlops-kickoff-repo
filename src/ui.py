import pandas as pd
import io
import logging
from nicegui import ui, events
from fastapi import FastAPI
from src.schemas import TelcoChurnInput
import numpy as np
# Vodafone-inspired colors
PRIMARY_RED = "#E60000"
BACKGROUND_WHITE = "#FFFFFF"
TEXT_DARK = "#333333"

logger = logging.getLogger(__name__)

def init(app: FastAPI):
    # State for batch processing
    state = {
        "batch_records": [],
        "df": None
    }

    @ui.page("/")
    def main_page():
        ui.query("body").style(f"background-color: {BACKGROUND_WHITE}; color: {TEXT_DARK}; font-family: 'Inter', sans-serif;")
        
        with ui.header().classes("items-center justify-between").style(f"background-color: {PRIMARY_RED}; color: white;"):
            ui.label("Vodafone Churn Prediction Portal").classes("text-2xl font-bold ml-4")
            with ui.row().classes("mr-4"):
                ui.button("Single Prediction", on_click=lambda: tabs.set_value("Single")).props("flat color=white")
                ui.button("Batch Prediction", on_click=lambda: tabs.set_value("Batch")).props("flat color=white")

        with ui.tabs().classes("w-full justify-center mt-4") as tabs:
            single_tab = ui.tab("Single")
            batch_tab = ui.tab("Batch")

        # Increased width of tab panels to max-w-6xl for a wider AGGrid
        with ui.tab_panels(tabs, value=single_tab).classes("w-full max-w-6xl mx-auto bg-transparent"):
            # --- SINGLE PREDICTION TAB ---
            with ui.tab_panel(single_tab):
                ui.markdown("### Single Customer Prediction").classes("text-center mb-4")
                
                # Form Grid
                with ui.grid(columns=2).classes("w-full gap-4"):
                    gender = ui.select(["Male", "Female"], label="Gender", value="Male").classes("w-full")
                    senior = ui.select([0, 1], label="Senior Citizen", value=0).classes("w-full")
                    partner = ui.select(["Yes", "No"], label="Partner", value="No").classes("w-full")
                    dependents = ui.select(["Yes", "No"], label="Dependents", value="No").classes("w-full")
                    tenure = ui.number(label="Tenure (months)", value=1, format="%.0f").classes("w-full")
                    phone = ui.select(["Yes", "No"], label="Phone Service", value="Yes").classes("w-full")
                    lines = ui.select(["No phone service", "No", "Yes"], label="Multiple Lines", value="No").classes("w-full")
                    internet = ui.select(["DSL", "Fiber optic", "No"], label="Internet Service", value="DSL").classes("w-full")
                    security = ui.select(["No internet service", "No", "Yes"], label="Online Security", value="No").classes("w-full")
                    backup = ui.select(["No internet service", "No", "Yes"], label="Online Backup", value="No").classes("w-full")
                    protection = ui.select(["No internet service", "No", "Yes"], label="Device Protection", value="No").classes("w-full")
                    support = ui.select(["No internet service", "No", "Yes"], label="Tech Support", value="No").classes("w-full")
                    tv = ui.select(["No internet service", "No", "Yes"], label="Streaming TV", value="No").classes("w-full")
                    movies = ui.select(["No internet service", "No", "Yes"], label="Streaming Movies", value="No").classes("w-full")
                    contract = ui.select(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month").classes("w-full")
                    billing = ui.select(["Yes", "No"], label="Paperless Billing", value="Yes").classes("w-full")
                    payment = ui.select(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check").classes("w-full")
                    monthly = ui.number(label="Monthly Charges", value=50.0).classes("w-full")
                    total = ui.input(label="Total Charges", value="50.0").classes("w-full")

                ui.button("Predict Churn", on_click=lambda: run_single_prediction()).classes("w-full mt-6 py-4 text-lg").style(f"background-color: {PRIMARY_RED}; color: white;")
                
                result_card = ui.card().classes("w-full mt-6 p-6 shadow-lg border-2")
                result_card.set_visibility(False)
                with result_card:
                    result_label = ui.label("").classes("text-3xl font-bold text-center w-full")
                    proba_label = ui.label("").classes("text-lg text-center w-full")

                async def run_single_prediction():
                    try:
                        data = {
                            "gender": gender.value,
                            "SeniorCitizen": float(senior.value),
                            "Partner": partner.value,
                            "Dependents": dependents.value,
                            "tenure": float(tenure.value),
                            "PhoneService": phone.value,
                            "MultipleLines": lines.value,
                            "InternetService": internet.value,
                            "OnlineSecurity": security.value,
                            "OnlineBackup": backup.value,
                            "DeviceProtection": protection.value,
                            "TechSupport": support.value,
                            "StreamingTV": tv.value,
                            "StreamingMovies": movies.value,
                            "Contract": contract.value,
                            "PaperlessBilling": billing.value,
                            "PaymentMethod": payment.value,
                            "MonthlyCharges": float(monthly.value),
                            "TotalCharges": float(total.value) if total.value and str(total.value).strip() else 0.0
                        }
                    
                        from src.api import predict
                        payload = [TelcoChurnInput(**data)]
                        response = predict(payload)
                        
                        res = response[0]
                        result_card.set_visibility(True)
                        if res.prediction == 1:
                            result_label.text = "CHURN PREDICTED"
                            result_card.style("border-color: #E60000; background-color: #FFF5F5;")
                            result_label.style("color: #E60000;")
                        else:
                            result_label.text = "NO CHURN PREDICTED"
                            result_card.style("border-color: #2D8A2D; background-color: #F5FFF5;")
                            result_label.style("color: #2D8A2D;")
                        
                        proba_label.text = f"Confidence: {res.proba*100:.1f}%" if res.proba else ""
                    except Exception as e:
                        logger.exception("Single prediction failed")
                        ui.notify(f"Prediction error: {str(e)}", type="negative")

            # --- BATCH PREDICTION TAB ---
            with ui.tab_panel(batch_tab):
                ui.markdown("### Batch Data Processing").classes("text-center mb-4")
                
                with ui.column().classes("w-full items-center gap-4"):
                    # Restricted to max-files=1 and auto_upload=True as requested
                    uploader = ui.upload(on_upload=lambda e: handle_upload(e), 
                                         on_rejected=lambda: ui.notify('File type not supported or already have a file!'), 
                                         label="1. Upload Data", 
                                         auto_upload=True).classes("w-full max-w-lg").props("accept=.csv,.xlsx max-files=1")
                    
                    

                # Batch Results Grid
                batch_results_grid = ui.aggrid({
                    'columnDefs': [],
                    'rowData': [],
                    'rowSelection': 'single',
                    'pagination': True,
                    'paginationPageSize': 20,
                    'defaultColDef': {'sortable': True, 'filter': True, 'resizable': True}
                }).classes("w-full mt-6 h-96 shadow-lg")
                batch_results_grid.set_visibility(False)

                generate_btn = ui.button("2. Generate Predictions", on_click=lambda: run_batch_prediction()).classes("w-full max-w-lg py-4").style(f"background-color: {PRIMARY_RED}; color: white;")
                generate_btn.set_visibility(False)

                async def handle_upload(e: events.UploadEventArguments):
                    try:
                        file = e.file
                        filename = file.name
                        ui.notify(f'file {file.name} uploaded', type="info")                        
                        
                        content_bytes = await file.read()

                        if filename.lower().endswith((".xlsx", ".xls")):
                            df = pd.read_excel(io.BytesIO(content_bytes))
                        elif filename.lower().endswith(".csv"):
                            df = pd.read_csv(io.BytesIO(content_bytes), sep=None, engine='python')
                        
                        state["batch_records"] = df.to_dict(orient="records")
                        state["df"] = df

                        # Restricted Columns List
                        requested_cols = [
                            ('prediction', 'Prediction'),
                            ('proba', 'Confidence'),
                            ('gender', 'Gender'),
                            ('tenure', 'Tenure'),
                            ('Contract', 'Contract'),
                            ('PaymentMethod', 'PaymentMethod'),
                            ('MonthlyCharges', 'MonthlyCharges'),
                            ('TotalCharges', 'TotalCharges')
                        ]

                        cols = []
                        for field, header in requested_cols:
                            col_def = {'headerName': header, 'field': field}
                            if field in ['prediction', 'proba']:
                                col_def['pinned'] = 'left'
                                if field == 'prediction':
                                    col_def['cellStyle'] = {'font-weight': 'bold'}
                            cols.append(col_def)

                        batch_results_grid.options['columnDefs'] = cols
                        
                        # Initial Preview
                        preview_data = []
                        for r in state["batch_records"]:
                            row = r.copy()
                            row["prediction"] = "WAITING..."
                            row["proba"] = 0.0
                            preview_data.append(row)
                            
                        batch_results_grid.options['rowData'] = preview_data
                        batch_results_grid.update()
                        
                        batch_results_grid.set_visibility(True)
                        generate_btn.set_visibility(True)
                        ui.notify(f"Successfully loaded {len(df)} records!", type="positive")
                        
                    except Exception as ex:
                        logger.exception("Upload handling failed")
                        ui.notify(f"Upload error: {str(ex)}", type="negative")

                async def run_batch_prediction():
                    if not state["batch_records"]:
                        ui.notify("No records to process!", type="warning")
                        return

                    try:
                        ui.notify("Generating predictions... please wait.")
                        from src.api import predict
                        
                        payload = [TelcoChurnInput(**r) for r in state["batch_records"]]
                        predictions = predict(payload)
                        
                        final_data = []
                        for i, p in enumerate(predictions):
                            row = state["batch_records"][i].copy()
                            row["prediction"] = "CHURN" if p.prediction == 1 else "STAY"
                            row["proba"] = float(p.proba) if p.proba is not None else 0.0
                            final_data.append(row)
                        
                        # Sort by probability descending
                        final_data.sort(key=lambda x: x["proba"], reverse=True)
                        
                        # Format for display
                        for r in final_data:
                            r["proba"] = f"{r['proba']*100:.1f}%"

                        batch_results_grid.options['rowData'] = final_data
                        batch_results_grid.update()
                        
                        ui.notify(f"Success! Predictions generated and sorted.", type="positive")
                    except Exception as ex:
                        logger.exception("Inference failed")
                        ui.notify(f"Inference error: {str(ex)}", type="negative")

    ui.run_with(app, title="Vodafone Churn Portal")
