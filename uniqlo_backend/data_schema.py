# Uniqlo Order Data Schema Definition
# ====================================
# This file defines the data schema for the Uniqlo Order Analysis System.
# All data fields are designed to accommodate real data from ERP/POS/Online platforms.
# 
# Related to Research: Chapter 3 - Data Sources and Collection
# 研究内容：多源订单数据采集与预处理

# ============================================================================
# ORDER DATA SCHEMA
# ============================================================================

ORDER_SCHEMA = {
    "Order_ID": {
        "type": "string",
        "description": "Unique order identifier (format: ORD-YYYYMMDD-XXXXXX)",
        "required": True,
        "example": "ORD-20240315-000001"
    },
    "Source": {
        "type": "string",
        "description": "Data source: ERP (线下门店), POS (门店POS), ONLINE_EC (线上电商平台)",
        "required": True,
        "enum": ["ERP", "POS", "ONLINE_EC", "APP", "MINI_PROGRAM"],
        "example": "POS"
    },
    "Region": {
        "type": "string",
        "description": "Sales region (province/city level)",
        "required": True,
        "example": "Shanghai"
    },
    "Store_Code": {
        "type": "string",
        "description": "Store identifier",
        "required": False,
        "example": "SH-001"
    },
    "Order_Date": {
        "type": "date",
        "description": "Order date (YYYY-MM-DD format)",
        "required": True,
        "example": "2024-03-15"
    },
    "Order_Time": {
        "type": "time",
        "description": "Order time (HH:MM:SS format)",
        "required": False,
        "example": "14:30:25"
    },
    "Season": {
        "type": "string",
        "description": "Season classification: Spring, Summer, Autumn, Winter",
        "required": True,
        "enum": ["Spring", "Summer", "Autumn", "Winter"],
        "example": "Spring"
    },
    "Category": {
        "type": "string",
        "description": "Product category",
        "required": True,
        "enum": [
            "T-Shirt", "Shirt", "Pants", "Jacket", "Coat", 
            "Dress", "Skirt", "Sweater", "Hoodie", "Innerwear",
            "Accessories", "Shoes", "Bags"
        ],
        "example": "T-Shirt"
    },
    "Product_Code": {
        "type": "string",
        "description": "Product SKU code",
        "required": True,
        "example": "SKU-123456"
    },
    "Product_Name": {
        "type": "string",
        "description": "Product name",
        "required": False,
        "example": "UT Crew Neck T-Shirt"
    },
    "Color": {
        "type": "string",
        "description": "Product color",
        "required": False,
        "example": "White"
    },
    "Size": {
        "type": "string",
        "description": "Product size",
        "required": False,
        "enum": ["XS", "S", "M", "L", "XL", "XXL", "Free"],
        "example": "M"
    },
    "Quantity": {
        "type": "integer",
        "description": "Number of items purchased",
        "required": True,
        "min": 1,
        "example": 2
    },
    "Unit_Price": {
        "type": "float",
        "description": "Unit price (CNY)",
        "required": True,
        "min": 0,
        "example": 99.00
    },
    "Discount": {
        "type": "float",
        "description": "Discount amount (CNY)",
        "required": False,
        "min": 0,
        "default": 0,
        "example": 10.00
    },
    "Total_Amount": {
        "type": "float",
        "description": "Total order amount after discount (CNY)",
        "required": True,
        "min": 0,
        "example": 188.00
    },
    "Payment_Method": {
        "type": "string",
        "description": "Payment method",
        "required": False,
        "enum": ["Cash", "Credit_Card", "Alipay", "WeChat_Pay", "Member_Card"],
        "example": "Alipay"
    },
    "Customer_ID": {
        "type": "string",
        "description": "Customer membership ID (if available)",
        "required": False,
        "example": "CUST-2024001"
    },
    "Age_Group": {
        "type": "string",
        "description": "Customer age group",
        "required": False,
        "enum": ["<18", "18-25", "26-35", "36-45", "46-55", ">55"],
        "example": "26-35"
    },
    "Gender": {
        "type": "string",
        "description": "Customer gender",
        "required": False,
        "enum": ["Male", "Female", "Unknown"],
        "example": "Male"
    }
}

# ============================================================================
# USER BEHAVIOR DATA SCHEMA (for K-means clustering)
# ============================================================================

USER_BEHAVIOR_SCHEMA = {
    "Customer_ID": {
        "type": "string",
        "description": "Unique customer identifier",
        "required": True,
        "example": "CUST-2024001"
    },
    "Total_Orders": {
        "type": "integer",
        "description": "Total number of orders",
        "required": True,
        "min": 0,
        "example": 15
    },
    "Total_Spend": {
        "type": "float",
        "description": "Total spending amount (CNY)",
        "required": True,
        "min": 0,
        "example": 5000.00
    },
    "Avg_Order_Value": {
        "type": "float",
        "description": "Average order value (CNY)",
        "required": True,
        "min": 0,
        "example": 333.33
    },
    "Purchase_Frequency": {
        "type": "float",
        "description": "Orders per month",
        "required": True,
        "min": 0,
        "example": 1.5
    },
    "Favorite_Category": {
        "type": "string",
        "description": "Most purchased category",
        "required": False,
        "example": "T-Shirt"
    },
    "Favorite_Season": {
        "type": "string",
        "description": "Most purchased season",
        "required": False,
        "example": "Summer"
    },
    "Preferred_Region": {
        "type": "string",
        "description": "Most frequently purchased region",
        "required": False,
        "example": "Shanghai"
    },
    "Last_Purchase_Date": {
        "type": "date",
        "description": "Last purchase date",
        "required": False,
        "example": "2024-03-01"
    },
    "Days_Since_Last_Purchase": {
        "type": "integer",
        "description": "Days since last purchase",
        "required": True,
        "min": 0,
        "example": 14
    }
}

# ============================================================================
# SALES TIME SERIES SCHEMA (for ARIMA)
# ============================================================================

SALES_TIMESERIES_SCHEMA = {
    "Date": {
        "type": "date",
        "description": "Date (YYYY-MM-DD)",
        "required": True,
        "example": "2024-03-15"
    },
    "Category": {
        "type": "string",
        "description": "Product category",
        "required": True,
        "example": "Coat"
    },
    "Region": {
        "type": "string",
        "description": "Sales region",
        "required": False,
        "example": "National"
    },
    "Sales_Volume": {
        "type": "integer",
        "description": "Daily sales volume",
        "required": True,
        "min": 0,
        "example": 1500
    },
    "Sales_Amount": {
        "type": "float",
        "description": "Daily sales amount (CNY)",
        "required": True,
        "min": 0,
        "example": 150000.00
    },
    "Promotion_Flag": {
        "type": "boolean",
        "description": "Whether there was a promotion",
        "required": False,
        "default": False,
        "example": True
    }
}

# ============================================================================
# TRANSACTION BASKET SCHEMA (for Apriori)
# ============================================================================

TRANSACTION_BASKET_SCHEMA = {
    "Transaction_ID": {
        "type": "string",
        "description": "Unique transaction identifier",
        "required": True,
        "example": "TXN-20240315-000001"
    },
    "Order_ID": {
        "type": "string",
        "description": "Associated order ID",
        "required": True,
        "example": "ORD-20240315-000001"
    },
    "Customer_ID": {
        "type": "string",
        "description": "Customer ID",
        "required": False,
        "example": "CUST-2024001"
    },
    "Transaction_Date": {
        "type": "date",
        "description": "Transaction date",
        "required": True,
        "example": "2024-03-15"
    },
    "Items": {
        "type": "array",
        "description": "List of product codes in this transaction",
        "required": True,
        "example": ["SKU-123456", "SKU-234567", "SKU-345678"]
    }
}

# ============================================================================
# REGIONAL SALES SCHEMA (for Decision Tree)
# ============================================================================

REGIONAL_SALES_SCHEMA = {
    "Region": {
        "type": "string",
        "description": "Sales region",
        "required": True,
        "example": "Shanghai"
    },
    "Season": {
        "type": "string",
        "description": "Season",
        "required": True,
        "example": "Winter"
    },
    "Category": {
        "type": "string",
        "description": "Product category",
        "required": True,
        "example": "Coat"
    },
    "Price_Range": {
        "type": "string",
        "description": "Price range: Low (<100), Mid (100-300), High (>300)",
        "required": True,
        "enum": ["Low", "Mid", "High"],
        "example": "Mid"
    },
    "Sales_Volume": {
        "type": "integer",
        "description": "Sales volume",
        "required": True,
        "min": 0,
        "example": 5000
    }
}
