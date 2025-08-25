
import streamlit as st
import sqlite3
import pandas as pd
import hashlib
import io
from datetime import datetime
from pydantic import BaseModel, field_validator

DB_PATH = 'refurbished.db'

USERS = {
  
    'admin': hashlib.sha256('admin123'.encode()).hexdigest(),
}

def check_login(username: str, password: str) -> bool:
    if not username or not password:
        return False
    hashed = hashlib.sha256(password.encode()).hexdigest()
    return USERS.get(username) == hashed

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS phones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            brand TEXT NOT NULL,
            model TEXT NOT NULL,
            storage TEXT,
            color TEXT,
            condition TEXT CHECK(condition IN ('New','Good','Scrap')) NOT NULL,
            cost_price REAL NOT NULL CHECK(cost_price >= 0),
            list_price REAL NOT NULL CHECK(list_price >= 0),
            stock INTEGER NOT NULL CHECK(stock >= 0),
            reserved_b2b INTEGER NOT NULL DEFAULT 0 CHECK(reserved_b2b >= 0),
            tags TEXT DEFAULT '',
            price_x REAL,
            price_y REAL,
            price_z REAL,
            override_x REAL,
            override_y REAL,
            override_z REAL,
            updated_at TEXT
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            phone_id INTEGER NOT NULL,
            platform TEXT CHECK(platform IN ('X','Y','Z')) NOT NULL,
            platform_condition TEXT,
            price REAL NOT NULL,
            fee REAL NOT NULL,
            net_revenue REAL NOT NULL,
            status TEXT CHECK(status IN ('SUCCESS','FAILED')) NOT NULL,
            reason TEXT,
            created_at TEXT,
            FOREIGN KEY(phone_id) REFERENCES phones(id)
        )
        """
    )
    conn.commit()
    conn.close()

# Optional schema example using Pydantic V2-style validators
class PhoneIn(BaseModel):
    condition: str
    tags: list[str]

    @field_validator("condition")
    @classmethod
    def validate_condition(cls, v):
        allowed = ["New", "As New", "Good", "Usable", "Scrap"]
        if v not in allowed:
            raise ValueError(f"Condition must be one of: {', '.join(allowed)}")
        return v

    @field_validator("tags")
    @classmethod
    def validate_tags(cls, v):
        if not isinstance(v, list):
            raise ValueError("Tags must be a list")
        return v


PLATFORMS = ['X','Y','Z']

FEE_STRUCTURES = {
    'X': lambda price: 0.10 * price,                # 10%
    'Y': lambda price: 0.08 * price + 2.0,          # 8% + $2
    'Z': lambda price: 0.12 * price                 # 12%
}

def map_condition_for_platform(condition: str, platform: str):
    condition = (condition or '').strip()
    if platform == 'X':
        mapping = {'New': 'New', 'Good': 'Good', 'Scrap': 'Scrap'}
        return mapping.get(condition, None), mapping.get(condition) is not None
    if platform == 'Y':
        mapping = {'New': '3 stars (Excellent)', 'Good': '2 stars (Good)', 'Scrap': '1 star (Usable)'}
        return mapping.get(condition, None), mapping.get(condition) is not None
    if platform == 'Z':
     
        mapping = {'New': 'New', 'Good': 'Good'}
        return mapping.get(condition, None), mapping.get(condition) is not None
    return None, False


def compute_suggested_prices(list_price: float) -> dict:
   
    prices = {}
    if list_price is None or list_price < 0:
        return {'X': None, 'Y': None, 'Z': None}
    prices['X'] = round((list_price + 0.0) / (1 - 0.10), 2)
    prices['Y'] = round((list_price + 2.0) / (1 - 0.08), 2)
    prices['Z'] = round((list_price + 0.0) / (1 - 0.12), 2)
    return prices


def available_stock(row) -> int:
    s = int(row['stock']) if row['stock'] is not None else 0
    r = int(row['reserved_b2b']) if row['reserved_b2b'] is not None else 0
    return max(0, s - r)


def upsert_phone(data: dict, phone_id: int | None = None):
    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()
    prices = compute_suggested_prices(data['list_price'])

    payload = (
        data['brand'].strip(),
        data['model'].strip(),
        data.get('storage','').strip(),
        data.get('color','').strip(),
        data['condition'].strip(),
        float(data['cost_price']),
        float(data['list_price']),
        int(data['stock']),
        int(data.get('reserved_b2b',0)),
        data.get('tags','').strip(),
        prices['X'], prices['Y'], prices['Z'],
        data.get('override_x'), data.get('override_y'), data.get('override_z'),
        now
    )

    if phone_id:
        cur.execute(
            """
            UPDATE phones SET brand=?, model=?, storage=?, color=?, condition=?, cost_price=?, list_price=?,
                stock=?, reserved_b2b=?, tags=?, price_x=?, price_y=?, price_z=?, override_x=?, override_y=?, override_z=?, updated_at=?
            WHERE id=?
            """,
            payload + (phone_id,)
        )
    else:
        cur.execute(
            """
            INSERT INTO phones (brand,model,storage,color,condition,cost_price,list_price,stock,reserved_b2b,tags,
                price_x,price_y,price_z,override_x,override_y,override_z,updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            payload
        )
    conn.commit()
    conn.close()


def delete_phone(phone_id: int):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("DELETE FROM phones WHERE id=?", (phone_id,))
    conn.commit()
    conn.close()


def list_phones_df(filters: dict | None = None) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query("SELECT * FROM phones", conn)
    conn.close()
    if df.empty:
        return df
    df['available_stock'] = df.apply(available_stock, axis=1)
    if filters:
        if (q := filters.get('q')):
            q = q.lower().strip()
            df = df[df['model'].str.lower().str.contains(q) | df['brand'].str.lower().str.contains(q)]
        if (cond := filters.get('condition')):
            df = df[df['condition'].isin(cond)]
    return df.reset_index(drop=True)


def get_phone(phone_id: int) -> dict | None:
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT * FROM phones WHERE id=?", (phone_id,))
    row = cur.fetchone()
    cols = [c[0] for c in cur.description] if cur.description else []
    conn.close()
    if not row:
        return None
    return dict(zip(cols, row))


def price_for_platform(row: dict, platform: str) -> float | None:
   
    override_key = {
        'X': 'override_x',
        'Y': 'override_y',
        'Z': 'override_z'
    }[platform]
    price_key = {
        'X': 'price_x',
        'Y': 'price_y',
        'Z': 'price_z'
    }[platform]
    return row.get(override_key) if row.get(override_key) not in (None, '') else row.get(price_key)


def attempt_listing(phone_id: int, platform: str):
    phone = get_phone(phone_id)
    if not phone:
        return False, 'Phone not found'

    # Stock check
    if available_stock(phone) <= 0:
        return False, 'Out of stock due to B2B/direct reservations'

    # Condition mapping and support
    mapped_cond, supported = map_condition_for_platform(phone['condition'], platform)
    if not supported:
        return False, f"Condition '{phone['condition']}' unsupported on platform {platform}"

    # Price determination
    price = price_for_platform(phone, platform)
    if price is None:
        return False, 'No price available for platform (check list price/overrides)'

    # Fee and profitability check
    fee = FEE_STRUCTURES[platform](price)
    net = price - fee
    margin = net - float(phone['cost_price'])
    if margin <= 0:
        # Avoid listing unprofitable phones
        log_listing(phone_id, platform, mapped_cond, price, fee, net, 'FAILED', 'Unprofitable after platform fees')
        return False, 'Listing blocked: unprofitable after platform fees'

    # Success path (mock)
    log_listing(phone_id, platform, mapped_cond, price, fee, net, 'SUCCESS', 'Mock listing successful')
    return True, 'Mock listing successful'


def log_listing(phone_id, platform, platform_condition, price, fee, net, status, reason):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO listings(phone_id,platform,platform_condition,price,fee,net_revenue,status,reason,created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
        """,
        (phone_id, platform, platform_condition, float(price), float(fee), float(net), status, reason, datetime.utcnow().isoformat())
    )
    conn.commit()
    conn.close()


def listings_df(filters: dict | None = None) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql_query(
        """
        SELECT l.id, l.created_at, l.platform, l.status, l.reason, l.price, l.fee, l.net_revenue,
               p.brand, p.model, p.condition, p.id as phone_id
        FROM listings l JOIN phones p ON p.id = l.phone_id
        ORDER BY l.id DESC
        """,
        conn
    )
    conn.close()
    if filters:
        if platform := filters.get('platform'):
            df = df[df['platform'].isin(platform)]
        if status := filters.get('status'):
            df = df[df['status'].isin(status)]
    return df.reset_index(drop=True)



def auth_gate():
    if 'auth' not in st.session_state:
        st.session_state.auth = False

    if st.session_state.auth:
        return True

    st.title('🔐 Login')
    with st.form('login_form', clear_on_submit=False):
        u = st.text_input('Username', key="login_username")
        p = st.text_input('Password', type='password', key="login_password")
        submitted = st.form_submit_button('Login')
        if submitted:
            if check_login(u.strip(), p):
                st.session_state.auth = True
                st.experimental_rerun()
            else:
                st.error('Invalid credentials')
    return False


def phone_form(existing: dict | None = None):
    condition = st.selectbox(
        'Condition*',
        ['New','Good','Scrap'],
        index=['New','Good','Scrap'].index((existing or {}).get('condition','New')),
        key=f"condition_{(existing or {}).get('id','new')}"
    )

    col1, col2 = st.columns(2)
    with col1:
        brand = st.text_input('Brand*', value=(existing or {}).get('brand',''), key=f"brand_{(existing or {}).get('id','new')}")
        model = st.text_input('Model*', value=(existing or {}).get('model',''), key=f"model_{(existing or {}).get('id','new')}")
        storage = st.text_input('Storage', value=(existing or {}).get('storage',''), key=f"storage_{(existing or {}).get('id','new')}")
        color = st.text_input('Color', value=(existing or {}).get('color',''), key=f"color_{(existing or {}).get('id','new')}")
    with col2:
        cost_price = st.number_input('Cost Price*', min_value=0.0, value=float((existing or {}).get('cost_price',0.0)), step=1.0, key=f"cost_{(existing or {}).get('id','new')}")
        list_price = st.number_input('Target Net Price (after fees)*', min_value=0.0, value=float((existing or {}).get('list_price',0.0)), step=1.0, key=f"list_{(existing or {}).get('id','new')}")
        stock = st.number_input('Stock*', min_value=0, value=int((existing or {}).get('stock',0)), step=1, key=f"stock_{(existing or {}).get('id','new')}")
        reserved_b2b = st.number_input('Reserved for B2B/Direct', min_value=0, value=int((existing or {}).get('reserved_b2b',0)), step=1, key=f"reserved_{(existing or {}).get('id','new')}")
        tags = st.text_input('Tags', value=(existing or {}).get('tags',''), key=f"tags_{(existing or {}).get('id','new')}")

    st.markdown('**Manual Price Overrides (optional)**')
    oc1, oc2, oc3 = st.columns(3)
    with oc1:
        override_x = st.number_input('Override X', min_value=0.0, value=float((existing or {}).get('override_x') or 0.0), step=1.0, key=f"ovx_{(existing or {}).get('id','new')}")
        use_ox = st.checkbox('Use override X', value=(existing or {}).get('override_x') is not None, key=f"use_ox_{(existing or {}).get('id','new')}")
        if not use_ox:
            override_x = None
    with oc2:
        override_y = st.number_input('Override Y', min_value=0.0, value=float((existing or {}).get('override_y') or 0.0), step=1.0, key=f"ovy_{(existing or {}).get('id','new')}")
        use_oy = st.checkbox('Use override Y', value=(existing or {}).get('override_y') is not None, key=f"use_oy_{(existing or {}).get('id','new')}")
        if not use_oy:
            override_y = None
    with oc3:
        override_z = st.number_input('Override Z', min_value=0.0, value=float((existing or {}).get('override_z') or 0.0), step=1.0, key=f"ovz_{(existing or {}).get('id','new')}")
        use_oz = st.checkbox('Use override Z', value=(existing or {}).get('override_z') is not None, key=f"use_oz_{(existing or {}).get('id','new')}")
        if not use_oz:
            override_z = None

    data = {
        'brand': brand,
        'model': model,
        'storage': storage,
        'color': color,
        'condition': condition,
        'cost_price': cost_price,
        'list_price': list_price,
        'stock': stock,
        'reserved_b2b': reserved_b2b,
        'tags': tags,
        'override_x': override_x,
        'override_y': override_y,
        'override_z': override_z,
    }
    return data


def validate_phone_payload(data: dict) -> list[str]:
    errs = []
    if not data['brand'].strip():
        errs.append('Brand is required')
    if not data['model'].strip():
        errs.append('Model is required')
    if data['cost_price'] < 0:
        errs.append('Cost price must be >= 0')
    if data['list_price'] < 0:
        errs.append('Target net price must be >= 0')
    if data['stock'] < 0:
        errs.append('Stock must be >= 0')
    if data['reserved_b2b'] < 0:
        errs.append('Reserved must be >= 0')
    if data['reserved_b2b'] > data['stock']:
        errs.append('Reserved cannot exceed stock')
    return errs

# -----------------------------
# Streamlit pages
# -----------------------------

def page_inventory():
    edit_id = st.number_input('Edit Phone ID', min_value=0, step=1, key="edit_phone_id")
    del_id = st.number_input('Delete Phone ID', min_value=0, step=1, key="delete_phone_id")

    q = st.text_input('Search (brand/model)', key="search_inventory")
    cond = st.multiselect('Filter by condition', ['New','Good','Scrap'], key="filter_condition_inventory")

    st.header('📦 Phone Inventory')
    st.caption('Add / Update / Delete phones. Prices for each platform are auto-suggested from Target Net Price.')

    with st.expander('➕ Add a new phone', expanded=False):
        data = phone_form()
        if st.button('Save Phone', type='primary', key="btn_save_phone"):
            errs = validate_phone_payload(data)
            if errs:
                for e in errs:
                    st.error(e)
            else:
                upsert_phone(data)
                st.success('Phone saved')
                st.experimental_rerun()

    st.subheader('Inventory List')
    df = list_phones_df({'q': q, 'condition': cond})

    if df.empty:
        st.info('No phones yet. Add one above or bulk upload.')
    else:
        # Show computed/suggested prices alongside overrides
        display = df.copy()
        display['use_price_x'] = display.apply(lambda r: price_for_platform(r, 'X'), axis=1)
        display['use_price_y'] = display.apply(lambda r: price_for_platform(r, 'Y'), axis=1)
        display['use_price_z'] = display.apply(lambda r: price_for_platform(r, 'Z'), axis=1)
        st.dataframe(
            display[['id','brand','model','storage','color','condition','cost_price','list_price','stock','reserved_b2b','available_stock','tags','use_price_x','use_price_y','use_price_z']],
            use_container_width=True
        )

        # Edit/Delete
        st.markdown('---')
        if st.button('Load for Edit', key="btn_load_for_edit"):
            if edit_id:
                row = get_phone(int(edit_id))
                if not row:
                    st.error('Phone ID not found')
                else:
                    st.write('Edit below and click **Update**')
                    data = phone_form(row)
                    if st.button('Update', type='primary', key="btn_update_phone"):
                        errs = validate_phone_payload(data)
                        if errs:
                            for e in errs:
                                st.error(e)
                        else:
                            upsert_phone(data, phone_id=int(edit_id))
                            st.success('Phone updated')
                            st.experimental_rerun()

        if st.button('Delete', type='secondary', key="btn_delete_phone"):
            if del_id:
                delete_phone(int(del_id))
                st.warning('Phone deleted')
                st.experimental_rerun()

    st.markdown('---')
    if st.button('🔁 Recompute Suggested Platform Prices for All', key="btn_recompute_all"):
        df = list_phones_df()
        if not df.empty:
            conn = get_conn()
            cur = conn.cursor()
            for _, r in df.iterrows():
                prices = compute_suggested_prices(float(r['list_price']))
                cur.execute(
                    "UPDATE phones SET price_x=?, price_y=?, price_z=?, updated_at=? WHERE id=?",
                    (prices['X'], prices['Y'], prices['Z'], datetime.utcnow().isoformat(), int(r['id']))
                )
            conn.commit(); conn.close()
            st.success('Suggested prices recomputed')
            st.experimental_rerun()


def page_bulk_upload():
    st.header('📤 Bulk Upload')
    st.caption('Upload CSV with headers: brand,model,storage,color,condition,cost_price,list_price,stock,reserved_b2b,tags,override_x,override_y,override_z')

    sample = pd.DataFrame([
        {
            'brand':'Apple','model':'iPhone 12','storage':'128GB','color':'Black','condition':'Good',
            'cost_price':350,'list_price':420,'stock':10,'reserved_b2b':2,'tags':'refurb',
            'override_x':'','override_y':'','override_z':''
        },
        {
            'brand':'Samsung','model':'Galaxy S21','storage':'256GB','color':'Silver','condition':'New',
            'cost_price':400,'list_price':520,'stock':5,'reserved_b2b':0,'tags':'hot',
            'override_x':'','override_y':'','override_z':''
        }
    ])
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    st.download_button('Download CSV Template', buf.getvalue(), file_name='bulk_template.csv', key="btn_download_template")

    file = st.file_uploader('Upload CSV', type=['csv'], key="csv_uploader")
    if file is not None:
        try:
            df = pd.read_csv(file)
            required = {'brand','model','condition','cost_price','list_price','stock'}
            if not required.issubset(set(df.columns)):
                st.error(f'Missing required columns: {sorted(list(required - set(df.columns)))}')
            else:
                # Clean and load rows
                count_ok = 0
                for _, row in df.fillna('').iterrows():
                    data = {
                        'brand': str(row['brand']).strip(),
                        'model': str(row['model']).strip(),
                        'storage': str(row.get('storage','')).strip(),
                        'color': str(row.get('color','')).strip(),
                        'condition': str(row['condition']).strip(),
                        'cost_price': float(row['cost_price']) if str(row['cost_price']).strip() != '' else 0.0,
                        'list_price': float(row['list_price']) if str(row['list_price']).strip() != '' else 0.0,
                        'stock': int(row['stock']) if str(row['stock']).strip() != '' else 0,
                        'reserved_b2b': int(row.get('reserved_b2b',0)) if str(row.get('reserved_b2b','')).strip() != '' else 0,
                        'tags': str(row.get('tags','')).strip(),
                        'override_x': float(row['override_x']) if str(row.get('override_x','')).strip() != '' else None,
                        'override_y': float(row['override_y']) if str(row.get('override_y','')).strip() != '' else None,
                        'override_z': float(row['override_z']) if str(row.get('override_z','')).strip() != '' else None,
                    }
                    errs = validate_phone_payload(data)
                    if errs:
                        st.warning(f"Skipping {data['brand']} {data['model']}: {errs}")
                        continue
                    upsert_phone(data)
                    count_ok += 1
                st.success(f'Imported {count_ok} phones')
                st.experimental_rerun()
        except Exception as e:
            st.exception(e)


def page_platforms():
    st.header('🧪 Dummy Platform Integrations')
    st.caption('Simulate listing a phone to platforms X, Y, Z with fee rules & condition mappings.')

    df = list_phones_df()
    if df.empty:
        st.info('Add phones in Inventory first.')
        return

    st.subheader('Attempt Listing')
    col1, col2, col3 = st.columns(3)
    with col1:
        phone_id = st.selectbox(
            'Phone',
            options=df['id'].tolist(),
            format_func=lambda i: f"#{i} - {df[df['id']==i]['brand'].values[0]} {df[df['id']==i]['model'].values[0]}",
            key="platform_phone_select"
        )
    with col2:
        platform = st.selectbox('Platform', options=PLATFORMS, key="platform_select")
    with col3:
        if st.button('List Now', type='primary', key="btn_list_now"):
            ok, msg = attempt_listing(int(phone_id), platform)
            if ok:
                st.success(msg)
            else:
                st.error(msg)

    st.markdown('---')
    st.subheader('Listing History & Filters')
    sel_plat = st.multiselect('Platform', PLATFORMS, key="history_filter_platforms")
    sel_status = st.multiselect('Status', ['SUCCESS','FAILED'], key="history_filter_status")

    ldf = listings_df({'platform': sel_plat, 'status': sel_status})
    if ldf.empty:
        st.info('No listings yet')
    else:
        st.dataframe(ldf, use_container_width=True)


def page_reports():
    st.header('📊 Profitability Checker (What-if)')
    df = list_phones_df()
    if df.empty:
        st.info('No phones to analyze')
        return
    st.caption('Shows net revenue and margin by platform using current prices (override > suggested).')

    rows = []
    for _, r in df.iterrows():
        for p in PLATFORMS:
            price = price_for_platform(r, p)
            if price is None:
                continue
            fee = FEE_STRUCTURES[p](price)
            net = price - fee
            margin = net - float(r['cost_price'])
            rows.append({
                'id': int(r['id']), 'brand': r['brand'], 'model': r['model'], 'condition': r['condition'],
                'platform': p, 'price': round(price,2), 'fee': round(fee,2), 'net': round(net,2), 'margin': round(margin,2)
            })
    rdf = pd.DataFrame(rows)
    if rdf.empty:
        st.info('No price data to show')
        return
    st.dataframe(rdf, use_container_width=True)


def page_settings():
    st.header('⚙️ Settings')
    st.caption('View current fee rules and condition mappings.')

    st.subheader('Fee Structures')
    st.write('- X: 10% fee')
    st.write('- Y: 8% fee + $2')
    st.write('- Z: 12% fee')

    st.subheader('Condition Mappings')
    cols = st.columns(3)
    with cols[0]:
        st.markdown('**Platform X**')
        st.write('New → New')
        st.write('Good → Good')
        st.write('Scrap → Scrap')
    with cols[1]:
        st.markdown('**Platform Y**')
        st.write('New → 3 stars (Excellent)')
        st.write('Good → 2 stars (Good)')
        st.write('Scrap → 1 star (Usable)')
    with cols[2]:
        st.markdown('**Platform Z**')
        st.write('New → New')
        st.write('Good → Good')
        st.write('Scrap → ❌ Unsupported (will fail)')

# -----------------------------
# Main app
# -----------------------------

def main():
    st.set_page_config(page_title='Refurbished Phone Manager', page_icon='📱', layout='wide')
    init_db()

    if not auth_gate():
        return

    with st.sidebar:
        st.title('📱 Refurbished Manager')
        st.success('Logged in as admin')
        page = st.radio('Navigate', ['Inventory','Bulk Upload','Platforms','Reports','Settings','Logout'], key="nav_radio")

    if page == 'Inventory':
        page_inventory()
    elif page == 'Bulk Upload':
        page_bulk_upload()
    elif page == 'Platforms':
        page_platforms()
    elif page == 'Reports':
        page_reports()
    elif page == 'Settings':
        page_settings()
    elif page == 'Logout':
        st.session_state.auth = False
        st.experimental_rerun()

if __name__ == '__main__':
    main()
