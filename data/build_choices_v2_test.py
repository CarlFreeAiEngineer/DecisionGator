#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Build data/choices-v2-test.jsonl. Each group: content plus list of (question, options, correct option, rationale).
The correct option is moved to the least-used index so correct indexes are spread evenly."""
import json, random, sys

SR, DR = "support_routing", "department_routing"
G = [
 (SR, "Hi, I'm so sorry to bother you, but could you please help me? I was charged twice for my March subscription and I'd really like one of the payments back.",
  [("Which team should handle this message?", ["support", "billing", "sales"], "billing", "A duplicate charge with a request to return one payment is a billing matter, even though it is worded as a plea for help.")]),
 (SR, "Please help! My card expired last week and I need to put the new one on my account before the next renewal goes through.",
  [("Where should this request be routed?", ["Customer support", "Payments", "Sales"], "Payments", "Replacing the card used for renewal payments is a payments task.")]),
 (SR, "Could someone help me get a copy of the receipt for order 55812? My manager needs it for my expense claim.",
  [("Which team should take this?", ["help desk", "finance", "sales", "shipping"], "finance", "Reissuing a receipt for an order is a billing document request, handled by finance.")]),
 (SR, "Hello, I need help adding our VAT number to the invoices you send us, since our accountant says it is missing.",
  [("Which team does this belong to?", ["support", "sales", "billing"], "billing", "Adding a tax number to invoices is a billing change.")]),
 (SR, "My automatic monthly payment failed on the 3rd and now my account says overdue. There's money in the account, so please help me sort out the payment.",
  [("Which team should pick up this ticket?", ["Customer support", "Billing", "Sales"], "Billing", "A failed recurring payment and an overdue balance are billing matters.")]),
 (SR, "I'm really hoping you can help me. I cancelled within the 14-day trial but was still charged $49, and I'd like that refunded please.",
  [("Who should deal with this?", ["help desk", "sales", "finance"], "finance", "A refund for a charge made after cancelling a trial is a finance matter.")]),
 (SR, "Can you help me figure out which plan would work for a team of 25? We're thinking of switching from our current provider next month.",
  [("Which team should reply?", ["support", "billing", "sales"], "sales", "Choosing a plan before buying is a sales question, even though it asks for help.")]),
 (SR, "I need some help: do you offer a discount for schools, and could you send a quote for 40 licences?",
  [("Which team should receive this message?", ["Sales", "Customer support", "Payments"], "Sales", "Discounts and a quote for new licences are pre-purchase sales questions.")]),
 (SR, "The app crashes every time I open the reports screen. I'm paying $30 a month for this, so please fix it soon.",
  [("Which team should handle this complaint?", ["billing", "support", "sales"], "support", "The problem is a crash; the monthly price is mentioned only for emphasis, not as a billing request.")]),
 (SR, "Since the last update, the invoice PDF download button does nothing when I click it. The invoice amount is fine, I just can't download it.",
  [("Which team should fix this?", ["billing", "sales", "support", "shipping"], "support", "A button that stops working after an update is a technical fault; the writer says the amount is correct.")]),
 (SR, "Your checkout page shows a blank white screen after I type my card details, so I can't finish buying the upgrade. No money has left my account.",
  [("Who should look into this first?", ["sales", "help desk", "finance"], "help desk", "A blank checkout page is a technical fault, and no charge was made.")]),
 (SR, "Two things, sorry: my login code never arrives by text, and I'd also like to know if you have an annual plan that's cheaper than paying monthly.",
  [("Which team should deal with the login code problem?", ["sales", "support", "billing"], "support", "A verification text that never arrives is a technical support issue."),
   ("Who should answer the question about an annual plan?", ["support", "billing", "sales"], "sales", "Which plans are offered and at what price is a sales question.")]),
 (SR, "Hi team, the tracking link for my order has shown 'label created' for nine days, and while you're at it, could you send me an invoice with my company name on it?",
  [("Which team should look into the stuck parcel?", ["help desk", "finance", "sales", "shipping"], "shipping", "A parcel that has not moved since its label was created is a shipping issue."),
   ("Which team should send the invoice with the company name?", ["shipping", "sales", "finance", "help desk"], "finance", "Issuing an invoice with different details is a finance task.")]),
 (SR, "I was billed for three seats but we only have two people now, and the calendar sync also stopped working yesterday.",
  [("Who should fix the sync?", ["Payments", "Customer support", "Sales"], "Customer support", "Calendar sync that stopped working is a technical support problem."),
   ("Which team should sort out the seat charge?", ["Sales", "Payments", "Customer support"], "Payments", "Being billed for more seats than are used is a payments question.")]),
 (SR, "Could you help me with two things? I'd like a refund for the headphones that arrived cracked, and I want to know whether the Pro model comes in white before I order one.",
  [("Which team should handle the refund?", ["sales", "billing", "support"], "billing", "Returning money for a damaged item is a billing task."),
   ("Who should answer the colour question?", ["billing", "support", "sales"], "sales", "Which colours a model comes in before ordering is a sales question.")]),
 (SR, "Please help. The smart plug won't connect to my Wi-Fi after the reset, and my receipt email shows the wrong billing address.",
  [("Which team should correct the receipt?", ["support", "billing", "sales", "shipping"], "billing", "A wrong billing address on a receipt is a billing correction."),
   ("Which team should help with the connection problem?", ["billing", "shipping", "support", "sales"], "support", "A device that will not join Wi-Fi is a technical support problem.")]),
 (SR, "Our renewal is due Friday. Before then, we'd like a quote for adding the analytics module, and we also need the card on file switched to our new corporate card.",
  [("Who should prepare the quote for the add-on?", ["billing", "sales", "support"], "sales", "Quoting a new module is a sales task."),
   ("Which team should change the card?", ["sales", "support", "billing"], "billing", "Changing the card on file is a billing task.")]),
 (SR, "I don't need anything technical and my bill is fine; I just want to talk to someone about upgrading to the enterprise tier.",
  [("Which team should this message go to?", ["support", "billing", "sales"], "sales", "The writer rules out technical and billing topics and asks about an upgrade, which is sales.")]),
 (SR, "Hello, I paid invoice 2207 by bank transfer two weeks ago but your system still shows it unpaid, and the password reset email goes to my old address.",
  [("Who should confirm the transfer was received?", ["help desk", "sales", "finance"], "finance", "Checking whether an invoice payment arrived is a finance task."),
   ("Who should help with the password reset email?", ["finance", "help desk", "sales"], "help desk", "Reset emails going to an old address is an account problem for the help desk.")]),
 (SR, "Please help, I'm desperate: the tax number on last quarter's invoices is wrong and our auditor needs corrected copies by Monday.",
  [("Which team should handle this?", ["Customer support", "Sales", "Billing", "Shipping"], "Billing", "Correcting the tax number on invoices is a billing task, however urgent the tone.")]),
 (SR, "Hi, hoping you can help. Could you send us copies of every invoice from 2025 in one file for our year-end accounts?",
  [("Which team should this be sent to?", ["support", "billing", "sales"], "billing", "Supplying copies of past invoices is a billing request.")]),
 (SR, "Hello, can you please help me understand what's included in the Business plan compared with Starter? I haven't signed up yet.",
  [("Which team should answer?", ["Payments", "Sales", "Customer support"], "Sales", "Comparing plans before signing up is a sales question.")]),
 (SR, "Could you please help? I was promised a 20% refund because my delivery arrived late, but it hasn't shown up on my card yet.",
  [("Which team should follow this up?", ["support", "sales", "billing"], "billing", "A promised refund that has not reached the card is a billing matter.")]),
 (SR, "The prices in the app show in US dollars even though I set my region to Canada. I haven't bought anything, I just think the setting isn't saving.",
  [("Which team should look at this?", ["billing", "support", "sales"], "support", "A region setting that does not save is a technical fault; no purchase or charge is involved.")]),
 (SR, "Each time I try to export my expense report to a spreadsheet, the totals column comes out empty. Can you help?",
  [("Where should this ticket go?", ["finance", "help desk", "sales"], "help desk", "An export that drops a column is a software problem, even though the data is about money.")]),
 (SR, "Could you help me get a demo scheduled for our operations team? We're comparing three vendors this month.",
  [("Which team should handle this request?", ["support", "sales", "billing", "shipping"], "sales", "A demo for a prospective buyer comparing vendors is a sales request.")]),
 (SR, "I need help with two things: your mobile app logs me out every few minutes, and I'd like a refund for the month I couldn't use it.",
  [("Who should handle the refund request?", ["support", "billing", "sales"], "billing", "A refund for a month of service is a billing request."),
   ("Who should fix the logouts?", ["billing", "sales", "support"], "support", "Being logged out repeatedly is a technical fault.")]),
 (SR, "Could you please help? The package with my order hasn't arrived after three weeks, and I've noticed the delivery fee was charged twice.",
  [("Which team should chase the missing parcel?", ["help desk", "finance", "sales", "shipping"], "shipping", "A parcel not delivered after three weeks is a shipping matter."),
   ("Which team should look at the double delivery fee?", ["shipping", "help desk", "finance", "sales"], "finance", "A fee charged twice is a billing matter for finance.")]),
 (SR, "Hi, I'm writing on behalf of our clinic. We'd like pricing for five more user accounts, and our current invoices need our new tax ID added.",
  [("Who should send the pricing?", ["Customer support", "Payments", "Sales"], "Sales", "Pricing for additional accounts is a sales question."),
   ("Who should update the invoices?", ["Sales", "Customer support", "Payments"], "Payments", "Adding a tax ID to invoices is a payments and billing change.")]),
 (SR, "Please could someone help me? My receipt says I paid 120 pounds but my order total was 102, so I think I was overcharged.",
  [("Which team should look into this?", ["sales", "billing", "support"], "billing", "A charge higher than the order total is a billing issue.")]),
 # department routing
 (DR, "My laptop won't connect to the office VPN since this morning, and I also noticed my overtime from last month is missing from my payslip.",
  [("Which department should handle the missing overtime?", ["HR", "IT", "Payroll", "Facilities"], "Payroll", "Overtime missing from a payslip is a payroll correction."),
   ("Which department should fix the VPN problem?", ["Payroll", "Facilities", "HR", "IT"], "IT", "A laptop that cannot connect to the VPN is an IT problem.")]),
 (DR, "The air conditioning in meeting room 4 is leaking onto the carpet, and I'd like to know how many days of parental leave I'm entitled to.",
  [("Who should deal with the leak?", ["IT", "Facilities", "HR", "Legal"], "Facilities", "A leaking air conditioner in a meeting room is a building issue for facilities."),
   ("Who should answer the parental leave question?", ["Facilities", "Legal", "IT", "HR"], "HR", "Leave entitlement is an HR policy question.")]),
 (DR, "A supplier sent us a contract with an indemnity clause I don't understand, and separately my building access card stopped working.",
  [("Which department should review the clause?", ["Legal", "Facilities", "HR", "Payroll"], "Legal", "Interpreting a contract clause is legal work."),
   ("Which department should sort out the access card?", ["HR", "Payroll", "Legal", "Facilities"], "Facilities", "Building access cards belong with facilities among these options.")]),
 (DR, "I'm a new starter and my bank details have changed, so my first salary should go to the new account. Also, my email password expired.",
  [("Where should the bank details change go?", ["Payroll", "HR", "IT", "General enquiries"], "Payroll", "Where salary is paid is a payroll detail."),
   ("Where should the password problem go?", ["General enquiries", "Payroll", "HR", "IT"], "IT", "An expired email password is an IT issue.")]),
 (DR, "Hello, I'm a student writing a school project and would like to know what year your company was founded.",
  [("Which department should reply?", ["HR", "Legal", "General enquiries", "IT"], "General enquiries", "A factual question from a member of the public fits general enquiries.")]),
 (DR, "I'd like to report that my manager keeps making comments about my age in team meetings.",
  [("Which department should receive this report?", ["Facilities", "HR", "IT", "Payroll"], "HR", "A complaint about a manager's conduct toward an employee goes to HR.")]),
 (DR, "The printer on the third floor keeps saying it's offline even though it's switched on and plugged in.",
  [("Which department should this go to?", ["Facilities", "IT", "HR"], "IT", "A powered printer showing offline is a network or device problem for IT.")]),
 (DR, "We received a letter saying another firm is using our trademark on their website, and we'd like advice on what to do.",
  [("Which department should handle this?", ["General enquiries", "Legal", "HR", "IT"], "Legal", "Trademark misuse is a legal matter.")]),
 (DR, "My tax code looks wrong on this month's payslip, and could someone please fix the broken lock on the second-floor bathroom door?",
  [("Which department should fix the lock?", ["Payroll", "Facilities", "IT", "HR"], "Facilities", "A broken door lock is building maintenance."),
   ("Which department should check the tax code?", ["Facilities", "IT", "Payroll", "HR"], "Payroll", "The tax code on a payslip is set by payroll.")]),
 (DR, "I start on the design team next week and need the drawing software installed on my laptop; I'd also like to update the emergency contact on my employee file.",
  [("Who should update the emergency contact?", ["IT", "HR", "Facilities", "Payroll"], "HR", "Employee file details such as emergency contacts are kept by HR."),
   ("Who should install the software?", ["HR", "Payroll", "IT", "Facilities"], "IT", "Installing software on a laptop is an IT task.")]),
 (DR, "Please help: I was paid for 30 hours this week but I worked 38.",
  [("Which department should this message go to?", ["HR", "Payroll", "General enquiries"], "Payroll", "Being underpaid for hours worked is a payroll correction.")]),
 (DR, "I took a call from a customer asking for our holiday opening hours. Also, the heating in the east wing has been off since Monday.",
  [("Which department should restore the heating?", ["Facilities", "HR", "General enquiries", "Payroll"], "Facilities", "Heating that is off is a building issue for facilities."),
   ("Where should the customer's question about opening hours go?", ["HR", "Facilities", "Payroll", "General enquiries"], "General enquiries", "A customer asking about opening hours is a general enquiry.")]),
 (DR, "Could you review the terms of the new office lease before we sign it on Thursday?",
  [("Which department should handle this request?", ["Facilities", "Legal", "Payroll"], "Legal", "Reviewing lease terms before signing is legal work, even though the lease is for an office.")]),
]

PROV = {"type": "synthetic", "generator": "AI assistant", "generation_recipe": "Individually authored English content, question, options and label; no model calls, no template expansion, no external examples copied.", "created": "2026-09-28"}
rng = random.Random(7)
counts = {}
out, n, fam = [], 0, {}
for gi, (family, content, qs) in enumerate(G, 1):
    for q, opts, correct, why in qs:
        assert correct in opts and len(set(opts)) == len(opts), q
        n += 1
        k = len(opts)
        best = min(counts.get(i, 0) for i in range(k))
        target = rng.choice([i for i in range(k) if counts.get(i, 0) == best])
        counts[target] = counts.get(target, 0) + 1
        others = [o for o in opts if o != correct]
        rng.shuffle(others)
        others.insert(target, correct)
        fam[family] = fam.get(family, 0) + 1
        out.append({"schema_version": 1, "id": f"choices-v2-test-{n:03d}", "group_id": f"choices-v2-test-grp-{gi:03d}",
                    "content": content, "question": q, "options": others, "label": target, "label_type": "choice",
                    "task_family": family, "source": "decisiongator-original-choices-v2-test", "provenance": PROV,
                    "license": "CC0-1.0", "rationale": why, "review_status": "synthetic_unreviewed", "split": "test"})
with open(sys.argv[1], "w", encoding="utf-8") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), fam, dict(sorted(counts.items())))
