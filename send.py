# import sys
# import requests
from datetime import datetime

from formatting import format_msg
# from send_mail import send_mail

def send(name, Website=None):#=None, to_email=None, verbose=False):
    msg = format_msg(my_name=name,my_Website=Website)
    #send  the message
    return msg
#     assert to_email != None
#     if website != None:
#         msg = format_msg(my_name=name,my_website=website)
#     else:
#         msg = format_msg(my_name=name)
#     if verbose:
#         print(name,website,to_email,to_email)
#     #send the message
#     send_mail(text=msg,to_emails=[to_email],html=None)
#
# if __name__ == "__main__":
#     print(sys.argv)
#     name = "Unknown"
#     if len(sys.argv) > 1:
#         name = sys.argv[1]
#     email = None
#     if len(sys.argv) > 2:
#         email = sys.argv[2]
#
#     response = send(name,to_email=email, verbose=True)
#     print(response)