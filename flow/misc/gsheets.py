from __future__ import print_function
import argparse
import httplib2
import os
import numpy as np
import pandas as pd

from apiclient import discovery
from oauth2client import client
from oauth2client import tools
from oauth2client.file import Storage
from googleapiclient.errors import HttpError

SCOPES = 'https://www.googleapis.com/auth/spreadsheets'
CLIENT_SECRET_FILE = 'client_secret.json'
APPLICATION_NAME = 'Andermann Lab'


def get_credentials():
    """Get valid user credentials from storage.

    If nothing has been stored, or if the stored credentials are invalid,
    the OAuth2 flow is completed to obtain the new credentials.

    Returns:
        Credentials, the obtained credential.
    """
    home_dir = os.path.expanduser('~')
    credential_dir = os.path.join(home_dir, '.credentials')
    if not os.path.exists(credential_dir):
        os.makedirs(credential_dir)
    credential_path = os.path.join(credential_dir,
                                   'sheets.googleapis.com-andermann-lab.json')

    store = Storage(credential_path)
    credentials = store.get()
    if not credentials or credentials.invalid:
        flow = client.flow_from_clientsecrets(CLIENT_SECRET_FILE, SCOPES)
        flow.user_agent = APPLICATION_NAME

        argparser = _create_argument_parser()
        flags = argparser.parse_args([])
        credentials = tools.run_flow(flow, store, flags)
        print('Storing credentials to ' + credential_path)
    return credentials


def append(sheet, page, values, retrys=3):
    """Append rows of data to a page."""
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discovery_url = ('https://sheets.googleapis.com/$discovery/rest?'
                     'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discovery_url)

    range_ = '{}!A:A'.format(page)

    sheets = service.spreadsheets()
    try_num = 1
    while try_num <= retrys:
        try:
            result = sheets.values().append(
                spreadsheetId=sheet,
                body={'range': range_, 'values': values, 'majorDimension': 'ROWS'},
                valueInputOption='USER_ENTERED', range=range_).execute()
        except HttpError:
            try_num += 1
        else:
            break

    assert result['updates']['updatedCells'] == np.prod(np.shape(values))


def clear(sheet, page, col_range='A:Z', retrys=3):
    """Clear an entire page."""
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discovery_url = ('https://sheets.googleapis.com/$discovery/rest?'
                     'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discovery_url)

    sheets = service.spreadsheets()
    range_ = '{}!{}'.format(page, col_range)
    try_num = 1
    while try_num <= retrys:
        try:
            result = sheets.values().clear(
                spreadsheetId=sheet, body={}, range=range_).execute()
        except HttpError:
            try_num += 1
        else:
            break

    return result


def read(sheet, page, range_, retrys=3):
    """Read a range of values from a page."""
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discovery_url = ('https://sheets.googleapis.com/$discovery/rest?'
                     'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discovery_url)

    range_name = '{}!{}'.format(page, range_)
    try_num = 1
    while try_num <= retrys:
        try:
            result = service.spreadsheets().values().get(
                spreadsheetId=sheet, range=range_name).execute()
        except HttpError:
            try_num += 1
        else:
            break

    values = result.get('values', [])
    return values

def dataframe(sheet, page, range_, retrys=3):
    """Convert the output of read into a pandas dataframe.

    :param sheet:
    :param page:
    :param range_:
    :param retrys:
    :return:

    """
    data = read(sheet, page, range_, retrys)
    colnames = [key.lower().replace('-', '_') for key in data[0]]
    ncols = len(colnames)
    df = pd.DataFrame(data[1:])
    addedcols = len(df.columns)
    df.columns = colnames[:addedcols]

    for i in range(addedcols, ncols):
        df[colnames[i]] = None

    return df


def _create_argument_parser():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--auth_host_name', default='localhost',
                        help='Hostname when running a local web server.')
    parser.add_argument('--noauth_local_webserver', action='store_true',
                        default=False, help='Do not run a local web server.')
    parser.add_argument('--auth_host_port', default=[8080, 8090], type=int,
                        nargs='*', help='Port web server should listen on.')
    parser.add_argument(
        '--logging_level', default='ERROR',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level of detail.')
    return parser


def get_pages(sheet, retrys=3):
    """Return names of all pages in a given spreadsheet."""
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discovery_url = ('https://sheets.googleapis.com/$discovery/rest?'
                     'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discovery_url)

    try_num = 1
    while try_num <= retrys:
        try:
            result = service.spreadsheets().get(
                spreadsheetId=sheet, includeGridData=False).execute()
        except HttpError:
            try_num += 1
        else:
            break
    sheets = [page['properties']['title'] for page in result['sheets']]
    return sheets


def main():
    """Shows basic usage of the Sheets API.

    Creates a Sheets API service object and prints the names and majors of
    students in a sample spreadsheet:
    https://docs.google.com/spreadsheets/d/1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms/edit
    """
    credentials = get_credentials()
    http = credentials.authorize(httplib2.Http())
    discoveryUrl = ('https://sheets.googleapis.com/$discovery/rest?'
                    'version=v4')
    service = discovery.build('sheets', 'v4', http=http,
                              discoveryServiceUrl=discoveryUrl)

    spreadsheetId = '1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms'
    rangeName = 'Class Data!A2:E'
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheetId, range=rangeName).execute()
    values = result.get('values', [])

    if not values:
        print('No data found.')
    else:
        print('Name, Major:')
        for row in values:
            # Print columns A and E, which correspond to indices 0 and 4.
            print('%s, %s' % (row[0], row[4]))


if __name__ == '__main__':
    main()
