msg_template = """"hello{name}, 
Thank you for joining {website}. we are very
happy to have you with us."""
#.format(name="justin",website='cfe.sh')

def format_msg(my_name="justin",my_Website = 'cfe.sh'):
    my_msg = msg_template.format(name=my_name,Website = my_Website)
    return my_msg



















# import smtplib
# from email.message import EmailMessage
#
#
# server = smtplib.SMTP("smtp.gmail.com", 587)
# server.starttls()
# server_login_mail = "pythonoynani@gmail.com"
# server_login_password = "vivek chary"
# server.login(server_login_mail, server_login_password)
#
#
# email = EmailMessage()
# email['From'] = server_login_mail
# email['To'] = "vivekchary757@gmail.com"
# email['subject']='Hii'
# email.set_content("helo")
# server.send_message(email)





# def say(text):
#     engine.say(text)
#     engine.runAndWait()
#
#
# say("hello sir, how can i help you? myself email assistant")
#
#
# # def assistant_listener():
# #     try:
# #         with sr.Microphone() as source:
# #             print("Listening...")
# #             voice = listener.listen(source)
# #             info = listener.recognize_google(voice, language="en-in").lower()
# #             return info
# #
# #     except:
# #         return "no"
#
#
# def send_email(rec, subject, message):
#     email = EmailMessage()
#     email['From'] = server_login_mail
#     email['To'] = rec
#     email['Subject'] = subject
#     email.set_content(message)
#     server.send_message(email)
#
#
# contact = {
#     "google": "google@gmail.com",
#     "youtube": "youtube@gmail.com"
# }
#
#
# def whattodo():
#     listen_me = assistant_listener()
#     if "assistant" in listen_me:
#         if "write mail" in listen_me:
#             say("To whom you want to send mail?")
#             try:
#                 user = assistant_listener()
#                 email = contact[user]
#             except:
#                 say(user+" not found in your contacts!")
#                 return 0
#             say("What you want to be subject?")
#             subject = assistant_listener()
#             say("what should be the message?")
#             message = assistant_listener()
#             send_email(email, subject, message)
#             say("Email Send Successfully")
#
# while True:
#     whattodo()