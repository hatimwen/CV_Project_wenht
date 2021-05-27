r"""
This py file is used to send e-mail automatically.
Generally, you should change the line from 15~21 according to yourself.
More details can be found on https://www.cnblogs.com/hatimwen/p/pythonmail.html
"""
import os
import smtplib
from email.mime.text import MIMEText
import argparse

parser = argparse.ArgumentParser(description='Mail Sender')
# model
parser.add_argument('--arch', '--a', type=str, metavar='ARCH', default='default net',
                    help='such as: 0520_se_resnext_3474_20_500')
parser.add_argument('--mailpath', '--mpath', type=str, metavar='PATH', default='mail')
parser.add_argument('--mailfile', '--mfile', type=str, metavar='FILE', default='mail_log.txt')
parser.add_argument('--mailhost', '--mhost', type=str, metavar='SMTP', default='smtp.163.com')
parser.add_argument('--mailusername', '--muser', type=str, metavar='NAME', default='hatimwen')
parser.add_argument('--mailauthkey', '--mkey', type=str, metavar='KEY', default='', help='the authorization code')
parser.add_argument('--mailsender', '--msender', type=str, metavar='SENDER', default='hatimwen@163.com')
parser.add_argument('--mailreceiver', '--mre', type=str, metavar='RE', default='')

if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    msg = args.arch
    mail_path = args.mailpath
    mail_file = args.mailfile
    mail_host = args.mailhost   # SMTP Server
    mail_user = args.mailusername
    # Password: (Not the login password of email , but the authorization code)
    mail_pass = args.mailauthkey
    # Sender's mail
    sender = args.mailsender
    # Receivers's mails
    receivers = [args.mailreceiver]   
    if not os.path.exists(mail_path):
        os.makedirs(mail_path)
    mail_log = open(os.path.join(mail_path, mail_file), 'a')
    
    content = 'Python Send Mail ! {}\'s training-process ends!'.format(msg)
    title = '{}\'s training-process ends'.format(msg)
    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = "{}".format(sender)
    message['To'] = ",".join(receivers)
    message['Subject'] = title
    
    try:
        smtpObj = smtplib.SMTP_SSL(mail_host, 465)
        smtpObj.login(mail_user, mail_pass)
        smtpObj.sendmail(sender, receivers, message.as_string())
        print("mail of {} has been send to {} successfully.".format(msg, receivers), file=mail_log)
        mail_log.flush()
    except smtplib.SMTPException as e:
        print(e, file=mail_log)
        mail_log.flush()
    mail_log.close()

