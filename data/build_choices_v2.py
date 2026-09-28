#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# ///
"""Build data/choices-v2.jsonl from individually authored groups below.

Each group: (family, content, [(question, options, rationale, answer), ...]).
The rationale is written before the answer. The answer is placed at a rotating index so labels spread evenly.
"""
import json
import sys
from collections import Counter
from pathlib import Path

S = 'support_routing'
R = 'request_type'
D = 'department_routing'

SUPPORT = [
    (S, "Hey, quick one. The checkout page throws an error 500 whenever I apply the promo code, and while you're at it, how much is the annual plan for a team of 12?", [
        ("Which team should fix the checkout error?", ["Technical support", "Billing", "Sales"], "An erroring checkout page is a software defect even though it is where payment happens.", "Technical support"),
        ("Who should answer the pricing question?", ["support", "sales", "billing"], "A price for a new team plan is a pre-purchase question for sales.", "sales")]),
    (S, "I'm so sorry to bother you, I'm not great with computers. Could you possibly help me? I stopped my membership last month but I was still charged $29.99 on the 14th.", [
        ("Which team should handle this?", ["support", "billing", "sales"], "Being charged after cancelling is a billing problem, even though it is phrased as a plea for help.", "billing"),
        ("Where should this ticket go?", ["Help desk", "Payments", "Account access", "Sales"], "A charge taken after the membership was stopped is a payments matter.", "Payments")]),
    (S, "Would it be possible to get a copy of the invoice for order 88213? Our accounts team needs it for the quarterly close.", [
        ("Which team should take care of this request?", ["Customer Support", "Billing", "Sales"], "Sending a copy of an invoice is a billing task, not general customer support.", "Billing"),
        ("Who needs to see this?", ["finance", "tech support"], "Invoice copies come from finance.", "finance")]),
    (S, "Hi there! We're a small bakery looking at your point-of-sale package and haven't signed up yet. Could you help us understand what the monthly cost would be with two card readers?", [
        ("Which team should take this?", ["support", "billing", "sales"], "A prospective customer asking about cost before buying is a sales lead; nothing has been billed.", "sales"),
        ("Route this message to:", ["Billing", "Sales", "Technical Support", "Shipping"], "The writer has not bought anything yet and wants a price, which sales provides.", "Sales")]),
    (S, "My card expires at the end of this month. Can someone help me put the new one on file so the subscription doesn't lapse?", [
        ("Which team should handle this?", ["Help desk", "Billing", "Sales"], "Changing the card used to pay for a subscription is a billing task.", "Billing"),
        ("Where does this belong?", ["tech support", "account access", "payments"], "Updating the payment card belongs with payments.", "payments")]),
    (S, "Help please!! The app logs me out every few minutes and I lose whatever I was typing. I pay for the premium plan so this is really frustrating.", [
        ("Which team should handle this?", ["billing", "support", "sales"], "The problem is a technical fault; mentioning the premium plan does not make it a billing issue.", "support"),
        ("Who needs to look at this?", ["Payments", "Technical support", "Account management"], "Being logged out repeatedly is a software fault for technical support.", "Technical support")]),
    (S, "Our company changed its registered address and VAT ID in July. Every invoice since then still shows the old details, so our auditor has flagged them. Could you reissue them?", [
        ("Which team should handle this?", ["Sales", "Customer support", "Billing"], "Reissuing invoices with corrected company and tax details is billing work.", "Billing"),
        ("Which department should reissue the documents?", ["Finance", "Tech support", "Sales", "Shipping"], "Invoices are finance documents, so finance reissues them.", "Finance")]),
    (S, "Is there any way you could help me out? I downgraded from Pro to Basic halfway through the month and I'd like the unused part of the Pro fee back.", [
        ("Which team should handle this?", ["support", "billing"], "Returning the unused part of a plan fee is a prorated refund, which billing handles.", "billing"),
        ("Which team should get this ticket?", ["Sales", "Technical support", "Billing", "Account access"], "The writer wants money back for a plan change, a billing matter.", "Billing")]),
    (S, "The payment page just spins forever after I press Pay. I've tried Firefox and Chrome. No money has been taken as far as I can tell.", [
        ("Which team should handle this?", ["Billing", "Tech support", "Sales"], "A payment page that hangs in two browsers is a technical fault, and no charge needs correcting.", "Tech support"),
        ("Which team has nothing to correct here, since no money was taken?", ["Billing", "Technical support"], "No charge was made, so there is nothing for billing to fix.", "Billing")]),
    (S, "Two quick things: I can't reset my password because the reset email never arrives, and I'd also like to know whether you offer a discount for nonprofits before we buy.", [
        ("Which team should help with the password problem?", ["Sales", "Account access", "Billing"], "A password reset that never arrives stops the writer getting into the account, an account access issue.", "Account access"),
        ("Which team should answer the discount question?", ["Account access", "Sales", "Billing"], "A discount question asked before buying is a sales question.", "Sales")]),
    (S, "Could you possibly look into something for me? My bank statement shows your company took £45 on 2 May and again on 3 May, but I only placed one order.", [
        ("Which team should handle this?", ["customer support", "billing", "sales"], "Being charged twice for one order is a billing error, despite the polite request for help.", "billing"),
        ("Which queue should this land in?", ["Help Desk", "Finance", "Shipping"], "A duplicate charge is for finance to investigate.", "Finance")]),
    (S, "Hi, I'd love some help choosing. We haven't signed up yet: is the Business tier worth it over Standard for a team of five, and can we pay yearly?", [
        ("Which team should take this?", ["Support", "Sales", "Billing"], "Comparing tiers and payment terms before signing up is a sales conversation; nothing has been billed.", "Sales"),
        ("Where should this ticket go?", ["account management", "tech support", "sales", "payments", "account access"], "The writer is not a customer yet, so this is a sales lead rather than account management.", "sales")]),
    (S, "Our monthly report export crashes at 90%. Separately, the last invoice was sent to a former employee's email; please send future invoices to our shared finance inbox instead.", [
        ("Which team should fix the export?", ["tech support", "billing", "sales", "shipping"], "An export that crashes is a technical defect.", "tech support"),
        ("Who should change where invoices are sent?", ["Technical support", "Billing", "Sales"], "The invoice delivery address is a billing setting.", "Billing")]),
    (S, "My order still hasn't arrived after 12 days and tracking hasn't moved since Tuesday. Can you help?", [
        ("Which team should handle this?", ["support", "shipping", "billing", "sales", "returns", "tech support"], "A delivery stuck in transit belongs to shipping, the specific team, rather than general support.", "shipping"),
        ("Who needs to see this?", ["Returns", "Shipping", "Payments"], "Nothing is being sent back; the parcel has not arrived, which is a shipping matter.", "Shipping")]),
    (S, "I sent the blender back two weeks ago and the courier confirmed delivery to your warehouse, but I haven't seen the money on my card yet.", [
        ("Which team should handle this?", ["Shipping", "Billing", "Sales", "Tech support"], "The return arrived, so the open issue is the missing refund, which billing handles.", "Billing"),
        ("Where should this ticket go?", ["support", "refunds and billing", "account access"], "A refund that has not reached the card is for refunds and billing, the specific team.", "refunds and billing")]),
    (S, "Hello, could you help? I need an itemised receipt for last Thursday's purchase so I can claim it on expenses.", [
        ("Which team should handle this?", ["Support", "Sales", "Billing"], "Issuing a receipt for a completed purchase is a billing task, not sales.", "Billing"),
        ("Who should this be sent to?", ["help desk", "finance"], "Receipts are finance documents.", "finance")]),
    (S, "I can't log in since I turned on two-factor and got a new phone. The subscription is paid through next year, so I don't want to lose it.", [
        ("Which team should handle this?", ["Billing", "Account access", "Sales"], "Being locked out after a phone change is an account access problem; the paid subscription is only context.", "Account access"),
        ("Where should this ticket go if only these teams exist?", ["billing", "tech support", "sales"], "Of these, a two-factor lockout is a technical support issue, not a payment one.", "tech support")]),
    (S, "Hi team, would it be possible to switch us from monthly card payments to annual invoicing paid by bank transfer? We're an existing customer on the Growth plan.", [
        ("Which team should handle this?", ["Sales", "Billing", "Support"], "Changing how an existing plan is paid, method and invoicing schedule, is a billing change.", "Billing"),
        ("Which team is this for?", ["Payments", "Technical Support"], "The request is about payment method and invoicing, which payments handles.", "Payments")]),
    (S, "Would you be able to tell me how much it would cost to add 30 more seats? We're already customers and want to expand before the new hires start.", [
        ("Which team should handle this?", ["billing", "account management", "support"], "Expanding an existing customer's seats is an upsell for account management, not a billing problem.", "account management"),
        ("Who should reply?", ["Sales", "Billing", "Technical support"], "A price for buying more seats is a sales question; nothing has been charged wrongly.", "Sales")]),
    (S, "The installer fails with 'license key invalid' even though we paid yesterday and got the confirmation email with the key.", [
        ("Which team should handle this?", ["Billing", "Technical support", "Sales"], "The payment went through and the key was issued, so a key the installer rejects is a technical problem.", "Technical support"),
        ("Which team does not need to get involved, since the payment went through?", ["Billing", "Tech support"], "The payment and confirmation arrived, so billing has nothing to fix.", "Billing")]),
    (S, "Please help, I'm desperate. My account was locked after too many wrong password attempts and I have a client demo in an hour.", [
        ("Which team should handle this?", ["Sales", "Account access", "Billing", "Support"], "A locked account is a specific account access problem.", "Account access"),
        ("Where should this go?", ["billing", "help desk", "sales"], "With no account access team listed, the help desk unlocks accounts; nothing here is about money.", "help desk")]),
    (S, "Could someone explain the 'platform fee' line on my March invoice? It wasn't there in February and nobody told me about it.", [
        ("Which team should handle this?", ["Customer support", "Sales", "Billing"], "Explaining a charge on an invoice is billing's job.", "Billing"),
        ("Who needs to see this?", ["tech support", "finance", "shipping", "sales"], "A question about an invoice line belongs to finance.", "finance")]),
    (S, "Hello! We're a school district evaluating tools for next year. Could you possibly send a quote for 400 student licences and tell us if you accept purchase orders?", [
        ("Where should this ticket go?", ["Billing", "Sales", "Support"], "A quote and purchase terms for a buyer who has not bought yet is sales, even though purchase orders sound financial.", "Sales"),
        ("Which team should handle this?", ["finance", "sales", "technical support", "account access"], "The district is evaluating before buying, so sales handles the quote.", "sales")]),
    (S, "I was charged in US dollars but my account is set to euros, and the currency switcher on the billing settings page shows a blank screen.", [
        ("Which team should fix the blank settings screen?", ["Billing", "Tech support", "Sales"], "A page that renders blank is a technical fault, even on a billing settings page.", "Tech support"),
        ("Which team should look at the currency of the charge?", ["Tech support", "Billing", "Sales"], "A charge taken in the wrong currency is a billing matter.", "Billing")]),
    (S, "Hiya, bit of a mess on my end. I bought the family plan by accident instead of the single one about ten minutes ago. Can I get switched and have the difference back?", [
        ("Which team should handle this?", ["support", "billing", "sales"], "Switching a just-bought plan and returning the price difference is a billing correction.", "billing"),
        ("Where should this go?", ["Payments", "Help Desk", "Shipping"], "Returning the price difference is a payments task.", "Payments")]),
    (S, "Our webhook calls started failing with a 401 after we renewed. The renewal went through fine and the invoice is paid.", [
        ("Which team should handle this?", ["billing", "technical support", "sales", "account management"], "Failing API calls are a technical problem; the renewal and invoice are fine.", "technical support"),
        ("Who should look at this?", ["Finance", "Engineering support", "Sales"], "An authentication error on webhooks is for engineering support.", "Engineering support")]),
    (S, "Can you help me understand why my trial turned into a paid plan? I don't think I ever entered a card, but there's a charge of $12.", [
        ("Which team should handle this?", ["Sales", "Billing", "Support"], "An unexpected charge after a trial is a billing question.", "Billing"),
        ("Where should this ticket go?", ["tech support", "payments"], "The customer is disputing a charge, which payments handles.", "payments")]),
    (S, "We've just become VAT registered. From next month on, all invoices you send us need to show our new registration number, which I've attached.", [
        ("Which team should handle this?", ["support", "sales", "billing"], "Adding a tax registration number to invoices is a billing change.", "billing"),
        ("Who needs to see this?", ["Account Access", "Finance", "Technical Support", "Sales", "Shipping"], "Invoice tax details are maintained by finance.", "Finance")]),
    (S, "I'd like to send the headphones back, they're uncomfortable. They're still unopened. What do I need to do?", [
        ("Which team should handle this?", ["support", "returns", "sales"], "Sending an unopened item back is handled by returns, the specific team.", "returns"),
        ("Where should this ticket go?", ["Billing", "Shipping", "Returns", "Tech support"], "The customer wants to start a return, not report a delivery or charge problem.", "Returns")]),
    (S, "The mobile app won't sync my notes since the last update, and I'd also like to know whether I can get an invoice addressed to my employer instead of me.", [
        ("Which team should deal with the sync problem?", ["billing", "support", "sales"], "Notes not syncing after an update is a technical problem for support.", "support"),
        ("Which team should deal with the invoice question?", ["support", "billing", "sales"], "Who an invoice is addressed to is a billing question.", "billing")]),
    (S, "Please could someone help? My direct debit bounced because I changed banks, and now there's a late fee on my account. I've set up the new bank details already.", [
        ("Which team should handle this?", ["Customer support", "Billing", "Sales"], "A bounced direct debit and a late fee are billing matters.", "Billing"),
        ("Who should this go to?", ["payments", "help desk", "account access"], "The bounced collection and fee belong with payments.", "payments")]),
    (S, "We run a clinic and want to know whether your scheduling system can import our existing patient calendar before we commit to buying.", [
        ("Which team should take this?", ["tech support", "sales", "billing"], "The clinic has not bought yet; questions from a prospective buyer go to sales.", "sales"),
        ("Who should reply to this?", ["Sales", "Customer Support"], "Support serves existing customers; a pre-purchase question is for sales.", "Sales")]),
    (S, "Charged 3 times for one pair of shoes!!! Order 5521. Fix this now.", [
        ("Which team should handle this?", ["support", "billing", "shipping", "sales", "returns", "account access"], "Three charges for one order is a billing error.", "billing"),
        ("Route this to:", ["Finance", "Returns", "Help desk"], "Repeated charges are for finance to investigate and reverse.", "Finance")]),
    (S, "The page where I'm supposed to download my invoices returns 'access denied', even though I'm the account owner.", [
        ("Which team should fix the error?", ["Billing", "Tech support", "Sales"], "The invoices themselves are fine; a page returning an error to its owner is a technical fault.", "Tech support"),
        ("Where should this ticket go?", ["Payments", "Technical support"], "An access error on a page is a technical support issue, not a payment one.", "Technical support")]),
    (S, "I love the new dashboard, great work. One problem: I was billed for 8 seats but we only have 6 users.", [
        ("Which team should handle the problem?", ["support", "billing", "sales"], "Being billed for more seats than used is a billing error.", "billing"),
        ("Where should this ticket go?", ["Account access", "Finance", "Technical support", "Sales"], "An overcharge for seats is for finance.", "Finance")]),
    (S, "Hi! Quick question before I sign up: do you charge extra for exporting to PDF, or is it included in the basic price?", [
        ("Which team should answer this?", ["billing", "sales", "support"], "A pricing question before signing up is for sales, not billing.", "sales"),
        ("Who should take this?", ["Payments", "Sales", "Tech Support", "Account Access"], "The writer is deciding whether to buy, so sales answers.", "Sales")]),
    (S, "Would it be possible to reset my two-factor device? I lost my phone and can't get into my dashboard at all.", [
        ("Which team should handle this?", ["support", "account access", "billing"], "Resetting two-factor to regain entry is an account access task.", "account access"),
        ("Where should this ticket go?", ["Sales", "Payments", "Help Desk"], "With no account access team listed, the help desk handles a lockout.", "Help Desk")]),
    (S, "Every time I try to update my card, the form says 'invalid postcode' even though it's correct. I've tried three times.", [
        ("Which team should handle this?", ["Billing", "Technical support", "Sales"], "The card is fine; a form rejecting a correct postcode is a technical defect.", "Technical support"),
        ("Who should fix the form?", ["payments", "tech support"], "Fixing a broken form is technical work.", "tech support")]),
    (S, "My parcel came, but inside was a blue kettle instead of the black one I paid for.", [
        ("Which team should handle this?", ["billing", "returns", "sales", "tech support"], "Swapping a wrong item is handled by returns; the payment itself is correct.", "returns"),
        ("Which team sent out the wrong item?", ["Shipping", "Billing", "Sales"], "Packing and sending the parcel is shipping's job.", "Shipping")]),
    (S, "Hello, I hope you can help. We were promised 20% off for the first year when we signed, but the invoice shows full price.", [
        ("Which team should correct the invoice?", ["Sales", "Billing", "Support"], "Correcting an invoice amount is billing's job, even though the discount came from sales.", "Billing"),
        ("Which team most likely made the promise the customer refers to?", ["Billing", "Sales", "Support"], "Discounts agreed at signing come from the sales conversation.", "Sales")]),
    (S, "Could you please send me a quote for adding the analytics add-on to our existing plan? Our renewal is in October.", [
        ("Which team should handle this?", ["billing", "account management", "tech support"], "Upselling an add-on to an existing customer is account management.", "account management"),
        ("Who needs to see this?", ["Sales", "Finance", "Help desk"], "A quote for an add-on is a sales request.", "Sales")]),
    (S, "Hi, would you mind helping me with something? I need last year's invoices re-sent as one PDF for our tax return.", [
        ("Which team should handle this?", ["support", "billing", "sales"], "Re-sending invoices is billing work, even when asked as a favour.", "billing"),
        ("Where should this ticket go?", ["finance", "tech support", "shipping"], "Invoices come from finance.", "finance")]),
    (S, "We want to buy another 10 licences for our existing account. Can you tell me how to do that?", [
        ("Which team should handle this?", ["Support", "Billing", "Sales"], "Buying more licences is a purchase, which sales handles.", "Sales"),
        ("Which team should take this?", ["tech support", "account management", "billing"], "Adding licences for an existing customer is account management.", "account management")]),
    (S, "Our website widget stopped loading on the pricing page yesterday. Customers can't see our prices at all.", [
        ("Which team should handle this?", ["Sales", "Technical support", "Billing"], "A widget that fails to load is a technical fault, even though it is on a pricing page.", "Technical support"),
        ("Who needs to see this?", ["billing", "sales", "help desk"], "The help desk handles a broken widget; no pricing or charge question was asked.", "help desk")]),
    (S, "I need to change the email I log in with, since I left my old job. Also, please stop sending the invoices to that old address.", [
        ("Which team should change the login email?", ["account access", "billing", "sales"], "The login email is an account access setting.", "account access"),
        ("Which team should change where invoices go?", ["Account access", "Billing", "Sales"], "The invoice recipient is a billing setting.", "Billing")]),
    (S, "The confirmation email says I paid for express delivery, but the package came by standard post after nine days. I'd like the express fee back.", [
        ("Which team should return the fee?", ["billing", "sales", "tech support"], "Paying back a delivery fee is a billing refund.", "billing"),
        ("Which team should look into why express delivery was not used?", ["Billing", "Shipping", "Sales"], "How the parcel was sent is a shipping question.", "Shipping")]),
    (S, "Hey, can you guys help? Invoice INV-2231 has the wrong company name on it and our accounts payable won't pay it until it's fixed.", [
        ("Which team should handle this?", ["support", "sales", "billing", "account access"], "Correcting an invoice is a billing task.", "billing"),
        ("Who should get this?", ["Finance", "Customer Support"], "An incorrect invoice is fixed by finance, not general support.", "Finance")]),
    (S, "I'm on the free version so I pay nothing, but the app drains my battery overnight even when it's closed.", [
        ("Which team should handle this?", ["billing", "support", "sales"], "Battery drain is a technical fault for support; there is no charge involved.", "support"),
        ("Which team is least relevant here?", ["Billing", "Technical support"], "The writer pays nothing, so billing is least relevant.", "Billing")]),
    (S, "Could you possibly tell me if there's a student discount? I'm thinking about subscribing next term.", [
        ("Which team should handle this?", ["support", "sales", "billing"], "A discount question from someone not yet subscribed is a sales question.", "sales"),
        ("Who should answer?", ["Customer support", "Finance", "Sales", "Shipping"], "The writer is a prospective buyer asking about price, so sales answers.", "Sales")]),
    (S, "I bought the annual plan by mistake, I meant to get monthly. Charged $240 about an hour ago. Please help!", [
        ("Which team should handle this?", ["support", "billing", "sales"], "Reversing a mistaken plan charge is billing work, despite the plea for help.", "billing"),
        ("Where should this go?", ["help desk", "payments", "account access", "shipping", "sales"], "The customer wants a charge undone, which is for payments.", "payments")]),
]

OTHER = [
    (R, "Hi there, I'm hoping you can help. The lamp I ordered flickers constantly and I'd rather just have my money back than try another one.", [
        ("What kind of request is this?", ["General question", "Refund", "Replacement"], "Wanting the money back is a refund request, whatever the polite framing.", "Refund"),
        ("What does the customer turn down?", ["Refund", "Replacement", "Repair"], "Preferring money back over trying another one declines a replacement.", "Replacement")]),
    (R, "Would it be possible to know whether your stores are open on public holidays?", [
        ("What type of request is this?", ["Complaint", "General enquiry", "Cancellation", "Booking"], "Asking about opening days is a general enquiry.", "General enquiry"),
        ("Which request type fits best?", ["Information", "Refund"], "The writer only wants information.", "Information")]),
    (R, "I've been on hold for 40 minutes three times this week and nobody has fixed my router. This is unacceptable and I want it escalated to a manager.", [
        ("What is the main purpose of the message?", ["Help request", "Complaint", "Sales enquiry"], "Calling the service unacceptable and demanding escalation makes this a complaint, not a plain help request.", "Complaint"),
        ("What does the writer want done with the case?", ["Escalation", "Cancellation", "Refund"], "The writer asks for it to go to a manager.", "Escalation")]),
    (R, "Could you help me please? I need to move my Friday 3pm fitting to any time next week.", [
        ("What does the customer want?", ["Help", "Reschedule", "Cancellation"], "Moving a fitting to another week is a reschedule; the fitting is kept.", "Reschedule"),
        ("How should this request be tagged?", ["Other", "Appointment change", "Complaint", "Refund"], "Changing the time of a fitting is an appointment change.", "Appointment change")]),
    (R, "Please close my account and delete my data. I'm not interested in a discount to stay.", [
        ("What is the customer requesting?", ["Account closure", "General support", "Discount"], "The customer asks for the account to be closed.", "Account closure"),
        ("What has the customer turned down?", ["Account closure", "A discount", "Data deletion"], "The customer says they do not want a discount to stay.", "A discount")]),
    (R, "Just letting you know the address on my profile is out of date, the new one is 14 Birch Lane. No rush.", [
        ("What kind of message is this?", ["Complaint", "Account update", "General question", "Urgent issue"], "Giving a new address for the profile is an account update, and the writer says it is not urgent.", "Account update"),
        ("Which category does this belong in?", ["Other", "Change of details"], "A new address is a change of details.", "Change of details")]),
    (R, "Can someone help? My order arrived with a cracked screen. Please send me a new one; I still want the phone.", [
        ("What is the customer asking for?", ["Help", "Replacement", "Refund"], "Asking for a new one sent is a replacement.", "Replacement"),
        ("Which request is the customer ruling out by saying they still want the phone?", ["Refund", "Repair", "Replacement", "Information"], "Still wanting the phone rules out giving it up for a refund.", "Refund")]),
    (R, "Hello, I was wondering if you might be able to credit back the delivery charge, since the parcel arrived five days late.", [
        ("What is the customer asking for?", ["Question", "Partial refund", "Complaint", "Replacement"], "Crediting back only the delivery charge is a partial refund.", "Partial refund"),
        ("Which request type best fits?", ["General enquiry", "Refund", "Delivery booking"], "Asking for a charge to be credited back is a refund request, not a general enquiry.", "Refund")]),
    (R, "Could you tell me what your refund policy is for digital downloads? I haven't bought anything yet.", [
        ("What is the writer asking for?", ["Refund", "Information", "Purchase"], "Asking about a policy before buying is a request for information.", "Information"),
        ("Which request type is ruled out because nothing has been bought?", ["Information", "Refund"], "With no purchase, there is nothing to refund.", "Refund")]),
    (R, "Any help would be appreciated: I'd like to cancel the gym membership from next month, and could you confirm the date of the final payment?", [
        ("What is the main request?", ["Help", "Cancellation", "Information"], "The main ask is to end the membership.", "Cancellation"),
        ("What is the secondary request?", ["Cancellation", "Information", "Refund"], "Confirming the final payment date is a request for information.", "Information")]),
    (D, "Hi, sorry to bother you, could you help me out? My last payslip is missing the overtime I worked on the 12th and 13th.", [
        ("Which department should handle this?", ["HR", "Payroll", "General enquiries"], "Missing overtime pay on a payslip is a payroll error.", "Payroll"),
        ("Who needs to see this?", ["IT", "Finance", "Facilities"], "Of these, pay corrections belong with finance.", "Finance")]),
    (D, "The heating on the third floor has been off since Monday and people are wearing coats at their desks.", [
        ("Which department should handle this?", ["Facilities", "IT", "General enquiries"], "Broken heating is a building problem for facilities.", "Facilities"),
        ("Where should this be sent?", ["HR", "Building maintenance", "Legal"], "Heating repairs are building maintenance.", "Building maintenance")]),
    (D, "Can someone help me? My laptop won't connect to the office Wi-Fi since the password change, so I can't submit my timesheet.", [
        ("Which department should handle this?", ["Payroll", "IT", "HR"], "The blocker is a Wi-Fi connection problem; the timesheet is only the consequence.", "IT"),
        ("Where should this ticket go?", ["Help desk", "Payroll", "Facilities"], "A laptop that cannot join the network is for the help desk.", "Help desk")]),
    (D, "I need to take four weeks of parental leave from March. Who do I talk to about the forms?", [
        ("Which department should handle this?", ["HR", "Payroll", "General enquiries", "Legal"], "Parental leave arrangements are handled by HR.", "HR"),
        ("Which team should get this?", ["People team", "Facilities", "IT"], "Leave forms go to the people team.", "People team")]),
    (D, "A supplier sent us a contract with an automatic renewal clause and a penalty for early exit. Can someone review it before I sign?", [
        ("Which department should review this?", ["Legal", "Procurement", "General enquiries"], "Reviewing contract clauses before signing is legal work.", "Legal"),
        ("Who needs to look at the contract?", ["Finance", "Legal", "IT", "HR"], "Contract terms and penalties are a legal review.", "Legal")]),
    (D, "My badge stopped opening the side door this morning, and I also wanted to ask whether the new pension contribution rate starts this month.", [
        ("Which department should fix the badge?", ["Facilities", "HR", "Payroll"], "Door access badges are managed by facilities.", "Facilities"),
        ("Which department should answer the pension question?", ["Facilities", "Payroll", "IT"], "Pension contributions come out of pay, so payroll answers.", "Payroll")]),
    (D, "Hello, would it be possible to be reimbursed for the train tickets from the client visit? I have the receipts.", [
        ("Which department should handle this?", ["General enquiries", "Finance", "HR"], "Reimbursing travel costs is finance work.", "Finance"),
        ("Which team should take this?", ["Expenses", "IT", "Facilities", "Legal"], "Travel reimbursement is an expenses claim.", "Expenses")]),
    (D, "I'd like to report that my manager keeps making comments about my accent in team meetings. I don't know who else to tell.", [
        ("Which department should handle this?", ["HR", "General enquiries", "Legal", "IT"], "A complaint about a manager's conduct goes to HR.", "HR"),
        ("Who should receive this report?", ["Facilities", "People and culture", "Payroll"], "Workplace conduct is handled by people and culture.", "People and culture")]),
    (D, "Could you possibly help? Someone left a box of printed files with customer names and addresses in the shared kitchen overnight.", [
        ("Who should be told about this?", ["Data protection officer", "Facilities", "General enquiries"], "Customer records left exposed is a data protection incident; the kitchen is only where it happened.", "Data protection officer"),
        ("Which department should handle this?", ["IT", "Compliance", "Payroll"], "Exposed personal data on paper is a compliance matter, not an IT one.", "Compliance")]),
    (D, "My tax code on the payslip looks wrong; I'm being taxed as if this is my second job.", [
        ("Which department should handle this?", ["HR", "Payroll", "Legal"], "Tax deductions on a payslip are payroll's responsibility.", "Payroll"),
        ("Where should this go?", ["General enquiries", "Finance", "IT"], "Of these, tax on pay belongs with finance.", "Finance")]),
    (D, "The printer on floor 2 jams on every double-sided job and keeps showing a network error.", [
        ("Which department should handle this?", ["IT", "Facilities", "General enquiries"], "A printer showing network errors is an IT fault.", "IT"),
        ("Who should look at the printer?", ["Tech support", "Office manager", "HR"], "Printer and network faults go to tech support.", "Tech support")]),
    (D, "I'm a visiting contractor and just need to know what time reception closes on Fridays.", [
        ("Who can answer this?", ["Reception", "HR", "IT"], "Reception knows its own opening hours.", "Reception"),
        ("Which department should handle this?", ["General enquiries", "Payroll", "Legal"], "A simple question about opening times is a general enquiry.", "General enquiries")]),
    (D, "Would you be able to help me? I haven't been paid for the last two weeks of my contract work, and the invoice I sent was acknowledged on the 2nd.", [
        ("Which department should handle this?", ["Payroll", "Accounts payable", "HR"], "A contractor paid by invoice is paid through accounts payable, not employee payroll.", "Accounts payable"),
        ("Where should this go?", ["General enquiries", "Finance", "IT"], "An unpaid supplier invoice is for finance.", "Finance")]),
    (D, "Our team needs three new monitors and a standing desk for the new hire starting Monday.", [
        ("Who should arrange the standing desk?", ["Facilities", "IT", "HR"], "Office furniture is arranged by facilities.", "Facilities"),
        ("Who should supply the monitors?", ["Facilities", "IT", "Payroll"], "Computer equipment such as monitors comes from IT.", "IT")]),
    (D, "I think I clicked a link in an email that asked for my login, and now there are strange sent messages in my outbox.", [
        ("Which department should handle this?", ["IT security", "General enquiries", "HR"], "A likely account compromise from a phishing link is an IT security incident.", "IT security"),
        ("Who needs to know right away?", ["Facilities", "IT", "Legal", "Payroll"], "A compromised email account is for IT.", "IT")]),
    (D, "Please could someone update my bank details for my salary? I've moved to a new bank.", [
        ("Which department should handle this?", ["HR", "Payroll", "General enquiries", "IT"], "The account salary is paid into is a payroll setting.", "Payroll"),
        ("Who should get this?", ["Finance", "Facilities"], "Salary payment details sit with finance.", "Finance")]),
    (D, "A former employee has written in asking for a copy of all the personal data we hold on her, citing data protection law.", [
        ("Which department should handle this?", ["Legal", "IT", "Payroll", "General enquiries"], "A formal data access request under the law is handled by legal.", "Legal"),
        ("Who should this be forwarded to?", ["Facilities", "Data protection officer", "Payroll"], "Personal data requests go to the data protection officer.", "Data protection officer")]),
    (D, "Help! The toilet on the ground floor is overflowing and water is spreading into the corridor.", [
        ("Which department should handle this?", ["Help desk", "Facilities", "General enquiries"], "A plumbing emergency is for facilities, despite the cry for help.", "Facilities"),
        ("Who should be called?", ["IT", "Maintenance", "HR", "Legal", "Payroll"], "Overflowing plumbing needs maintenance.", "Maintenance")]),
    (D, "I've been offered a job elsewhere and want to know how much notice I have to give under my contract.", [
        ("Which department should handle this?", ["HR", "Legal", "Payroll", "General enquiries"], "Notice periods in an employment contract are an HR question.", "HR"),
        ("Who should answer this?", ["Facilities", "People team", "IT"], "Employment terms are answered by the people team.", "People team")]),
    (D, "Two things: the VPN drops every half hour when I work from home, and my holiday allowance shows 5 days fewer than I was given in my contract.", [
        ("Which department should fix the VPN?", ["IT", "HR", "Facilities"], "A dropping VPN connection is an IT fault.", "IT"),
        ("Which department should correct the holiday allowance?", ["IT", "HR", "Payroll"], "Leave entitlement records are maintained by HR.", "HR")]),
]

PROVENANCE = {"type": "synthetic", "generator": "AI assistant", "generation_recipe": "Individually authored English content, question, options and label; no model calls, no template expansion, no external examples copied.", "created": "2026-09-28"}


def interleave(a, b):
    out, ia, ib = [], 0, 0
    total = len(a) + len(b)
    for k in range(total):
        # keep the running share of b close to its overall share
        if ib < len(b) and (ia >= len(a) or (ib + 1) * total <= (k + 1) * len(b) + len(b) // 2):
            out.append(b[ib]); ib += 1
        else:
            out.append(a[ia]); ia += 1
    return out


def split_for(n):
    m = n % 20
    return 'validation' if m in (0, 1) else 'calibration' if m == 2 else 'train'


def main(out_path):
    groups = interleave(SUPPORT, OTHER)
    counters = Counter()
    rows, rec = [], 0
    for g, (family, content, items) in enumerate(groups, 1):
        for question, options, rationale, answer in items:
            assert options.count(answer) == 1, (question, answer)
            assert len(set(o.casefold() for o in options)) == len(options), options
            n = len(options)
            target = (counters[n] * 3 + g) % n  # spreads labels without a fixed cycle
            counters[n] += 1
            i = options.index(answer)
            shift = (i - target) % n
            placed = options[shift:] + options[:shift]
            assert placed[target] == answer
            rec += 1
            rows.append({"schema_version": 1, "id": f"choices-v2-{rec:03d}", "group_id": f"choices-v2-grp-{g:03d}",
                         "content": content, "question": question, "options": placed, "label": target,
                         "label_type": "choice", "task_family": family, "source": "decisiongator-original-choices-v2",
                         "provenance": PROVENANCE, "license": "CC0-1.0", "rationale": rationale,
                         "review_status": "synthetic_unreviewed", "split": split_for(g)})
    Path(out_path).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
    print('records', len(rows), 'groups', len(groups))
    print('splits', Counter(r['split'] for r in rows), 'groups', Counter(split_for(g) for g in range(1, len(groups) + 1)))
    print('families', Counter(r['task_family'] for r in rows))
    print('labels', Counter(r['label'] for r in rows), 'sizes', Counter(len(r['options']) for r in rows))
    for s in ('validation', 'calibration'):
        print(s, Counter(r['task_family'] for r in rows if r['split'] == s))


if __name__ == '__main__':
    main(sys.argv[1])
