base_features = [
    "Type_enc",
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]"
]
target_column = "Machine failure9"


extended_features = (
    base_features
    + [
        "ambient_temp_C",
        "factory_load_pct"
    ]
)
print("Base Features:", len(base_features))
print("Extended Features:", len(extended_features))