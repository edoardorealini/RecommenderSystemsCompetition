import requests
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def mail_to(toaddr, subject, text):
	fromaddr = "iPadAvailabilityChecker@gmail.com"

	msg = MIMEMultipart()
	msg['From'] = fromaddr
	msg['To'] = toaddr
	msg['Subject'] = subject

	body = text

	msg.attach(MIMEText(body, 'plain'))

	server = smtplib.SMTP('smtp.gmail.com', 587)
	server.ehlo()
	server.starttls()
	server.ehlo()
	server.login("softwarenotificationalert@gmail.com", "test.1234")
	text = msg.as_string()
	server.sendmail(fromaddr, toaddr, text)