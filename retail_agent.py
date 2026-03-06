# Retail AI Assistant — Personal Shopper + Customer Support Agent
from dotenv import load_dotenv
load_dotenv()

import ast 
import pandas as pd
from datetime import datetime
import os

# load data
from pathlib import Path
DATA_DIR = Path(__file__).resolve().parent # its just load data from anywhere from same folder

products = pd.read_csv(DATA_DIR / "product_inventory.csv")

# need bit cleaning bez pandas is a bit dumb it treats almost everything that isn't a simple number as a plain string (text). so agent read data properly bcz these 3 columns are in string format
products["sizes_available"] = products["sizes_available"].apply(lambda x: [int(s) for s in str(x).split("|")]) # Before: "8|10|12" (just text) , After: [8, 10, 12] (List of integers)
products["stock_per_size"] = products["stock_per_size"].apply(lambda x: {int(k): int(v) for k, v in ast.literal_eval(str(x)).items()}) # Before: "{'8': 5, '10': 2}" (useless text), After: {8: 5, 10: 2} (a real Dictionary). ast.literal_eval (a safe way to turn string-code into real code) 
products["tags"] = products["tags"].apply(lambda x: [t.strip().lower() for t in str(x).split(",")]) # Before: "evening,formal,red" (just text) , After: ['evening', 'formal', 'red'] (List of strings)

# same pandas problems You cannot do math on strings, but you can do math on DateTime objects.
orders = pd.read_csv(DATA_DIR / "orders.csv")
orders["order_date"] = pd.to_datetime(orders["order_date"]) # Before "2026-01-28" (Text/String) , After: 2026-01-28 00:00:00 (Actual Date)

with open(DATA_DIR / "policy.txt", "r") as f:
    policy_text = f.read()


# ── model ──────────────────────────────────────────────────
from langchain.chat_models import init_chat_model
llm = init_chat_model(model="qwen/qwen3-32b", model_provider="groq")

# from langchain_openai import ChatOpenAI
# llm = ChatOpenAI(
#     model="gpt-5"
# )


# tools
from langchain_core.tools import tool


# "Search the CSV with filters like price, size, occasion and return matching products"
@tool
def search_products(occasion: str = "", max_price: float = 0, size: int = 0, prefer_sale: bool = False, style_tags: str = "") -> str:
    """Search products by filters. Use this when customer wants product recommendations.

    Args:
        occasion: e.g. 'evening', 'cocktail', 'bridal', 'prom'
        max_price: max budget, 0 = no limit
        size: dress size (2-16), 0 = any
        prefer_sale: True to prioritize sale items
        style_tags: comma-separated like 'modest,lace,sleeve'
    """
    df = products.copy() # its just make a copy of orginal data 

    # Block 1: "What is the customer looking for?"  eg. Customer says: "evening, modest" This collects all the style words into one list ['evening', 'modest'], then removes every dress that doesn't have at least one of those tags.
    tags_to_match = []
    if occasion:
        tags_to_match.append(occasion.strip().lower())
    if style_tags:
        tags_to_match.extend([t.strip().lower() for t in style_tags.split(",")])
    if tags_to_match:
        df = df[df["tags"].apply(lambda t: any(tag in t for tag in tags_to_match))]

    # Block 2: "What's the budget?" eg. Customer says: "under $300".throw out everything above $300.
    if max_price > 0:
        df = df[df["price"] <= max_price]

    # Block 3: "Does it come in their size AND is it in stock?"  Customer says: "size 8" This checks TWO things at once: Does this dress come in size 8? Is it actually in stock (not 0)?
    if size > 0:
        df = df[df["stock_per_size"].apply(lambda s: s.get(size, 0) > 0)]

    if df.empty:
        return "No products found. Try relaxing some filters."

    # Block 4: "Sort the best ones first"
    df = df.sort_values(by=["is_sale", "bestseller_score"] if prefer_sale else ["bestseller_score"], ascending=False)

    # Block 5: "Write a nice summary for each dress" -> eg. Take the top 10 dresses and format each one into a clean line with all the important info — product ID, name, price, tags, stock, sale status.
    results = []
    for _, r in df.head(10).iterrows():
        stock_info = f", Stock(size {size}): {r['stock_per_size'].get(size, 'N/A')}" if size > 0 else ""
        sale_info = f", ON SALE (was ${r['compare_at_price']})" if r["is_sale"] else ""
        clearance = ", CLEARANCE-FINAL SALE" if r["is_clearance"] else ""
        results.append(f"• {r['product_id']} | {r['title']} | ${r['price']} | {', '.join(r['tags'])} | Score: {r['bestseller_score']}{stock_info}{sale_info}{clearance}")

    return f"Found {len(df)} products:\n" + "\n".join(results)


# "Look up one specific product by its ID"
@tool
def get_product(product_id: str) -> str:
    """Get details of a product by ID. Use when customer asks about a specific product.

    Args:
        product_id: e.g. 'P0001'
    """
    row = products[products["product_id"] == product_id.strip().upper()] #  looks through the whole CSV for a product_id that matches exactly what the customer said. -> strip() makes sure there are no accidental spaces. upper() makes sure p0042 and P0042 both work.
    if row.empty:
        return f"ERROR: Product '{product_id}' not found."
    r = row.iloc[0]
    stock = ", ".join([f"Size {s}: {q}" for s, q in sorted(r["stock_per_size"].items())]) # Before: {8: 5, 10: 2}, After: "Size 8: 5, Size 10: 2"
    return f"{r['product_id']} | {r['title']} | ${r['price']} (was ${r['compare_at_price']}) | {r['vendor']} | Tags: {', '.join(r['tags'])} | Sale: {r['is_sale']} | Clearance: {r['is_clearance']} | Score: {r['bestseller_score']} | Stock: {stock}"

# "Look up one specific order by its ID"
@tool
def get_order(order_id: str) -> str:
    """Fetch order details. Use when customer asks about an order or before evaluating return.

    Args:
        order_id: e.g. 'O0001'
    """
    row = orders[orders["order_id"] == order_id.strip().upper()]
    if row.empty:
        return f"ERROR: Order '{order_id}' not found in our system."
    o = row.iloc[0]
    p = products[products["product_id"] == o["product_id"]] #  this like does cross varify from order and product_intventory at very top i store a in global variable
    product_info = ""
    if not p.empty:
        p = p.iloc[0]
        product_info = f" | Product: {p['title']} | Vendor: {p['vendor']} | Sale: {p['is_sale']} | Clearance: {p['is_clearance']}"
    return f"Order {o['order_id']} | Date: {o['order_date'].strftime('%Y-%m-%d')} | Product: {o['product_id']} | Size: {o['size']} | Paid: ${o['price_paid']} | Customer: {o['customer_id']}{product_info}"

# "Check if this order can be returned based on the policy rules"
@tool
def evaluate_return(order_id: str) -> str:
    """Evaluate if an order is eligible for return based on policy. Always use this for return questions.

    Args:
        order_id: e.g. 'O0001'
    """
    # Block 1: "Find the Order"
    row = orders[orders["order_id"] == order_id.strip().upper()]
    if row.empty:
        return f"ERROR: Order '{order_id}' not found. Cannot evaluate."

    # Block 2: "Cross-reference the Product"
    o = row.iloc[0]
    p = products[products["product_id"] == o["product_id"]]
    if p.empty:
        return f"ERROR: Product '{o['product_id']}' not found."
    p = p.iloc[0]

    # Block 3: "How Many Days Ago?" (bcz llm have cutoff knowledge and here llm only depends upon tool outputs)
    days = (datetime.now() - o["order_date"]).days

    # clearance = final sale
    if p["is_clearance"]:
        return f"❌ DENIED | Clearance item — final sale, no returns/exchanges. (Order {order_id}, {days} days ago)"

    # vendor: Aurelia Couture = exchanges only
    if p["vendor"] == "Aurelia Couture":
        window = 7 if p["is_sale"] else 14
        if days <= window:
            return f"⚠️ EXCHANGE ONLY | Aurelia Couture policy — no refunds, exchanges only. Within {window}-day window ({days} days). Customer pays return shipping."
        return f"❌ DENIED | Aurelia Couture exchanges only, and {window}-day window expired ({days} days)."

    # vendor: Nocturne = 21-day window
    if p["vendor"] == "Nocturne":
        if p["is_sale"]:
            if days <= 7:
                return f"✅ APPROVED (Store Credit) | Nocturne sale item, within 7-day window ({days} days)."
            elif days <= 21:
                return f"⚠️ EXCHANGE ONLY | Nocturne sale item, past 7 days but within 21-day extended window ({days} days)."
            return f"❌ DENIED | Past Nocturne's 21-day window ({days} days)."
        if days <= 21:
            return f"✅ APPROVED (Full Refund) | Nocturne extended 21-day window ({days} days)."
        return f"❌ DENIED | Past Nocturne's 21-day window ({days} days)."

    # sale items = 7 days, store credit
    if p["is_sale"]:
        if days <= 7:
            return f"✅ APPROVED (Store Credit) | Sale item, within 7-day window ({days} days)."
        return f"❌ DENIED | Sale item, 7-day window expired ({days} days)."

    # normal items = 14 days, full refund
    if days <= 14:
        return f"✅ APPROVED (Full Refund) | Normal item, within 14-day window ({days} days)."
    return f"❌ DENIED | Normal item, 14-day window expired ({days} days)."


# system prompt
system_prompt = f"""You are a Retail AI Assistant — Personal Shopper + Customer Support.

SHOPPING: Use search_products to find items. Never invent products.
SUPPORT: Use get_order then evaluate_return for returns. Never guess policy.
If an ID is not found, say so. Never hallucinate.

Return Policy:
{policy_text}
"""


# agent
from langchain.agents import create_agent

agent = create_agent(
    model=llm,
    tools=[search_products, get_product, get_order, evaluate_return],
    system_prompt=system_prompt,
)


# chat loop
from langchain.messages import HumanMessage

print("🛍️ Retail AI Assistant is ready! Type 'exit' to quit.\n")

chat_log = []  # collect chat messages to save later
run_count = 0   # counter for langsmith run names

try:
    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() in {"exit", "quit", "q"}:
            print("👋 Goodbye!")
            break

        run_count += 1
        response = agent.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config={"recursion_limit": 15, "run_name": f"retail_agent_run_{run_count}"}
        )
        ai_reply = response["messages"][-1].content
        print("AI:", ai_reply, "\n")

        chat_log.append(f"You: {user_input}")
        chat_log.append(f"AI: {ai_reply}\n")

except KeyboardInterrupt:
    print("\n👋 Bye!")

# save chat to demo.txt on exit
if chat_log:
    with open(DATA_DIR / "demo.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(chat_log))
    print("💾 Chat saved to demo.txt")
