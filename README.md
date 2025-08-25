# Refurbished Phone App  

This is a small app I built to **manage refurbished phone inventory**.  
It lets you add, edit, delete, and bulk upload phone details using a clean interface.  

---

## What it does  

- Add new phone records (brand, model, storage, etc.).  
- Keep track of **stock** and **reserved** units.  
- Upload a whole CSV file of phones in one go.  
- Makes sure data is clean (e.g., condition must be `New`, `Good`, or `Scrap`).  

---

##  Setup  

1. Clone or download this repo.  
   ```bash
   git clone https://github.com/os019/phoneapp.git
   cd phoneapp-app
   ```

2. (Optional but recommended) Create a virtual environment:  
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows  
   source venv/bin/activate   # Mac/Linux  
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ How to run  

```bash
streamlit run app.py
```

This will open the app in your browser.  

---

## Bulk Upload (CSV format)  

If you have lots of phones, you can upload them using a CSV file.  

### Required columns  
- `brand`  
- `condition` → only `New`, `Good`, or `Scrap` are allowed  
- `model`  
- `storage`  
- `color`  
- `cost_price`  
- `list_price`  
- `stock`  

###  Optional columns  
- `reserved` (defaults to 0)  
- `tags` (comma separated, e.g., `Bestseller, Premium`)  
- `override_x`, `override_y`, `override_z`  

### Example  

```csv
brand,condition,model,storage,color,cost_price,list_price,stock,reserved,tags,override_x,override_y,override_z
Apple,New,iPhone 13,128GB,Black,45000,50000,10,2,"Bestseller, Premium",48000,49000,49500
Samsung,Good,Galaxy S21,256GB,Silver,25000,30000,5,1,"Budget, Clearance",27000,28000,29000
Xiaomi,Scrap,Redmi Note 10,64GB,Gray,8000,10000,15,5,"Budget",9000,9500,9700
```

---

## Common issues  

- **IntegrityError: condition IN ('New','Good','Scrap')**  
  → Means you entered an invalid condition. Fix it in the CSV.  

- **DuplicateWidgetID: st.selectbox**  
  → Happens if two Streamlit widgets have the same key. Just make the keys unique.  

---

##  Made by  

Onkar Uddhav Sugave  
 Pune, India  
 sugaveonkar@gmail.com  
