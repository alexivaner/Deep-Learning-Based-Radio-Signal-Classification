"""
   Copyright (C) 2020, Foundation for Research and Technology - Hellas
   This software is released under the license detailed
   in the file, LICENSE, which is located in the top-level
   directory structure 
"""

from PIL import Image
from tqdm import tqdm

import argparse
import requests
import urllib
import os
import errno
import paramiko
import getpass as pswd


class SatnogsDownloader(argparse.Action):
    """A class to facilitate downloading SatNOGS data"""

    def __init__(
            self, observation_id, ground_station, satellite_norad_cat_id,
            obs_start, obs_end, transmitter, vetted_status_list, vetted_user,
            dest, obs_num, user, password, host, port):
        self.root = 'https://network.satnogs.org/api/observations/'
        self.observation_id = observation_id
        self.ground_station = ground_station
        self.satellite_norad_cat_id = satellite_norad_cat_id
        self.obs_start = obs_start
        self.obs_end = obs_end
        self.transmitter = transmitter
        self.vetted_status_list = vetted_status_list
        self.vetted_user = vetted_user
        self.__init_destination_path(dest)
        self.obs_num = obs_num
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def __init_destination_path(self, dest_path):
        if (not os.path.exists(dest_path)):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT),
                                    dest_path)
        self.dest = os.path.join(dest_path, '')

    def _set_vetted_status(self, status):
        self.vetted_status = status

    def _get_endpoint(self, args):
        print('https://network.satnogs.org/api/observations/' + args)
        return 'https://network.satnogs.org/api/observations/' + args

    def _get_observations(
            self, page):
        args = '?page=' + str(page)
        args += '&end=' + str(self.obs_end)
        args += '&start=' + str(self.obs_start)
        args += '&ground_station=' + str(self.ground_station)
        args += '&satellite__norad_cat_id=' + str(self.satellite_norad_cat_id)
        args += '&id=' + str(self.observation_id)
        args += '&transmitter=' + str(self.transmitter)
        # TODO: Consider vetter_status parameter to optimize query
        args += '&vetted_status=' + str(self.vetted_status)
        args += '&vetted_user=' + str(self.vetted_user)
        return requests.get(self._get_endpoint(args))

    def crop(self, image_path, coords, saved_location):
        """
        @param image_path: The path to the image to edit
        @param coords: A tuple of x/y coordinates (x1, y1, x2, y2)
        @param saved_location: Path to save the cropped image
        """
        image_obj = Image.open(image_path)
        cropped_image = image_obj.crop(coords)
        cropped_image.save(saved_location)

    '''
    Function that transfers raw IQ SatNOGS observations that match specific
    criteria from a remote to a local server.
    '''
    def transfer_raw_observations(self):

        host = self.host
        port = self.port
        transport = paramiko.Transport((host, port))

        # Auth
        password = self.password
        username = self.user
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        filepath = '/spare/iq/leonidio/iq_s16_'
        localpath = self.dest + '/iq_s16_'
        for s in self.vetted_status_list:
            self._set_vetted_status(s)
            p = 1
            cnt = 0
            finished = False

            while (cnt < int(self.obs_num) or finished):
                resp = self._get_observations(p)
                if resp.status_code == 200:
                    data = resp.json()
                elif resp.status_code == 404:
                    finished = True

                for w in range(1, min(len(data), int(self.obs_num)-cnt+1)):
                    filepath = filepath + str(data[w]["id"]) + ".dat"
                    localpath = localpath + str(data[w]["id"]) + ".dat"
                    try:
                        sftp.get(filepath, localpath)
                    except FileNotFoundError:
                        print("File not found, but oh well...")
                    filepath = '/spare/iq/leonidio/iq_s16_'
                    localpath = self.dest + '/iq_s16_'
                    cnt += 1
                p = p + 1
            sftp.close()
            transport.close()

    def download_waterfall_plot(self):

        for s in self.vetted_status_list:
            self._set_vetted_status(s)
            p = 1

            pbar = tqdm(total=int(self.obs_num))
            finished = False
            cnt = 0
            while (cnt < int(self.obs_num) or finished):

                resp = self._get_observations(p)
                if resp.status_code == 200:
                    data = resp.json()
                elif resp.status_code == 404:
                    finished = True

                for w in range(1, min(len(data), int(self.obs_num)-cnt+1)):
                    if data[w]["waterfall"] is None:
                        continue

                    if data[w]["vetted_status"] in self.vetted_status:
                        path = self.dest + '/' + data[w]["vetted_status"]
                        try:
                            os.makedirs(path)
                        except OSError as e:
                            if e.errno != errno.EEXIST:
                                raise
                        urllib.urlretrieve(
                            data[w]["waterfall"], path + '/tmp.png')
                        self.crop(path + '/tmp.png', (106, 26, 650, 1524),
                                  path + '/' + data[w]["vetted_status"] +
                                  '_' + str(cnt) + '.png')
                        cnt += 1
                        pbar.update()
                        os.remove(path + '/tmp.png')
                p = p + 1
            pbar.close()


def parse_vetted_status(value):
    return value.split(",")


def argument_parser():
    description = 'A tool to facilitate the downloading of SatNOGS data.'
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=description)
    parser.add_argument(
        "-g --ground-station", dest="ground_station", default="",
        help="Set ground-station (Default: %(default)s)")
    parser.add_argument(
        "-i --id", dest="observation_id", default="",
        help="Set observation id (Default: %(default)s)")
    parser.add_argument(
        "-c --satellite-norad-cat-id", dest="satellite_norad_cat_id",
        default="", help="Set NORAD category ID (Default: %(default)s)")
    parser.add_argument(
        "-s --observation-start", dest="obs_start", default="",
        help="Set observation start (Default: %(default)s)")
    parser.add_argument(
        "-e --observation-end", dest="obs_end", default="",
        help="Set observation end (Default: %(default)s)")
    parser.add_argument(
        "-t --transmitter", dest="transmitter", default="",
        help="Set satellite transmitter (Default: %(default)s)")
    parser.add_argument(
        "-v --vetted-status", dest="vetted_status_list",
        default='good', type=parse_vetted_status,
        help="Set the vetted status of the observation \
            (Default: %(default)s)")
    parser.add_argument(
        "-u --vetted-user", dest="vetted_user", default="",
        help="Set the vetter of the observation (Default: %(default)s)")
    parser.add_argument(
        "-d --destination", dest="dest", required=True,
        help="Set the destination directory for downloaded waterfalls \
            (Default: %(default)s)")
    parser.add_argument(
        "-n --observation-num", dest="obs_num", required=True,
        help="Set the number of observations per vetting status to download \
            (Default: %(default)s)")
    parser.add_argument(
        '--host', dest='host', required=True, action="store",
        help="Set the sftp host")
    parser.add_argument(
        '--port', dest='port', required=True, action="store",
        help="Set the sftp port", type=int) 
    parser.add_argument(
        '--user', dest='user', required=True, help="Set the sftp username")
    parser.add_argument(
        '--password', dest='password', required=True, action="store_true",
        help="Set the sftp password")
    parser.add_argument(
        '--transfer-iq', dest='transfer', action="store_true", default="False",
        help="Set the sftp password")
    return parser


def main(downloader=SatnogsDownloader, args=None):
    if args is None:
        args = argument_parser().parse_args()

    d = downloader(
        observation_id=args.observation_id,
        ground_station=args.ground_station,
        satellite_norad_cat_id=args.satellite_norad_cat_id,
        obs_start=args.obs_start, obs_end=args.obs_end,
        transmitter=args.transmitter,
        vetted_status_list=args.vetted_status_list,
        vetted_user=args.vetted_user, dest=args.dest,
        obs_num=args.obs_num,
        user=args.user,
        password=pswd.getpass(),
        host=args.host,
        port=args.port)

    if args.transfer:
        d.transfer_raw_observations()


if __name__ == '__main__':
    main()
