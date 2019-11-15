#!/usr/bin/env python
# coding: utf-8

# finsta360 - python script to finalize incomplete MP4 of Insta360 ONE-X
# https://github.com/kichiki/finsta360

import sys
import os.path

import struct
from datetime import datetime, timedelta
import gc
#from tqdm import tqdm


# ## parsing mp4

def parse_mvhd(buf):
    # Movie Header Atoms

    version            = buf[0]
    creation_time      = struct.unpack('>I', buf[4:8])[0]
    modification_time  = struct.unpack('>I', buf[8:12])[0]
    time_scale         = struct.unpack('>I', buf[12:16])[0]
    duration           = struct.unpack('>I', buf[16:20])[0]
    preferred_rate     = struct.unpack('>I', buf[20:24])[0]
    preferred_volume   = struct.unpack('>H', buf[24:26])[0]
    # next 10 bytes are reserved
    matrix_structure   = struct.unpack('>IIIIIIIII', buf[36:72])
    preview_time       = struct.unpack('>I', buf[72:76])[0]
    preview_duration   = struct.unpack('>I', buf[76:80])[0]
    poster_time        = struct.unpack('>I', buf[80:84])[0]
    selection_time     = struct.unpack('>I', buf[84:88])[0]
    selection_duration = struct.unpack('>I', buf[88:92])[0]
    current_time       = struct.unpack('>I', buf[92:96])[0]
    next_track_id      = struct.unpack('>I', buf[96:100])[0]

    print(f'version            : {version}')
    print(f'creation time      : {datetime(1904,1,1) + timedelta(seconds=creation_time)}')
    print(f'modification_time  : {datetime(1904,1,1) + timedelta(seconds=modification_time)}')
    print(f'time scale         : {time_scale}')
    print(f'duration           : {duration} / {duration/time_scale} sec / {duration/time_scale/60} min')
    print(f'preferred_rate     : {preferred_rate}')
    print(f'preferred_volume   : {preferred_volume}')
    print(f'matrix_structure   : {matrix_structure}')
    print(f'preview_time       : {preview_time}')
    print(f'preview_duration   : {preview_duration}')
    print(f'poster_time        : {poster_time}')
    print(f'selection_time     : {selection_time}')
    print(f'selection_duration : {selection_duration}')
    print(f'current_time       : {current_time}')
    print(f'next_track_id      : {next_track_id}')

def parse_tkhd(buf):
    #Track Header Atoms

    version   = buf[0]
    flags     = buf[1:4]
    creation_time      = struct.unpack('>I', buf[4:8])[0]
    modification_time  = struct.unpack('>I', buf[8:12])[0]
    track_id           = struct.unpack('>I', buf[12:16])[0]
    # next 4 bytes are reserved
    duration           = struct.unpack('>I', buf[20:24])[0]
    # next 8 bytes is reserved
    layer              = struct.unpack('>H', buf[32:34])[0]
    alternate_group    = struct.unpack('>H', buf[34:36])[0]
    volume             = struct.unpack('>H', buf[36:38])[0]
    # next 2 bytes are reserved
    matrix_structure   = struct.unpack('>IIIIIIIII', buf[40:76])
    track_width        = struct.unpack('>I', buf[76:80])[0]
    track_height       = struct.unpack('>I', buf[80:84])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'creation time     : {datetime(1904,1,1) + timedelta(seconds=creation_time)}')
    print(f'modification_time : {datetime(1904,1,1) + timedelta(seconds=modification_time)}')
    print(f'track_id          : {track_id}')
    print(f'duration          : {duration}')
    print(f'layer             : {layer}')
    print(f'alternate_group   : {alternate_group}')
    print(f'volume            : {volume}')
    print(f'matrix_structure  : {matrix_structure}')
    print(f'track_width       : {track_width}')
    print(f'track_height      : {track_height}')

def parse_mdhd(buf):
    #Media Header Atoms

    version   = buf[0]
    flags     = buf[1:4]
    creation_time      = struct.unpack('>I', buf[4:8])[0]
    modification_time  = struct.unpack('>I', buf[8:12])[0]
    time_scale         = struct.unpack('>I', buf[12:16])[0]
    duration           = struct.unpack('>I', buf[16:20])[0]
    language           = struct.unpack('>H', buf[20:22])[0]
    quality            = struct.unpack('>H', buf[22:24])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'creation time     : {datetime(1904,1,1) + timedelta(seconds=creation_time)}')
    print(f'modification_time : {datetime(1904,1,1) + timedelta(seconds=modification_time)}')
    print(f'time scale        : {time_scale}')
    print(f'duration          : {duration} / {duration/time_scale} sec / {duration/time_scale/60} min')
    print(f'language          : {language}')
    print(f'quality           : {quality}')


def parse_stsd(buf):
    #Sample Description Atoms

    print('DATA:')
    print_binaries(buf)
    print(f'size of buf: {len(buf)}')

    version   = buf[0]
    flags     = buf[1:4]
    n_entries = struct.unpack('>I', buf[4:8])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'number of entries : {n_entries}')

    sample_description_table = []
    for i in range(n_entries):
        i0 = 8 + i*4
        i1 = i0 + 4
        if len(buf) < i1: break
        sample_description_size = struct.unpack('>I', buf[i0:i0+4])[0]
        data_format = str(buf[i0+4:i0+8], 'utf-8')
        data_reference_index = struct.unpack('>H', buf[i0+14:i0+16])[0]
        sample_description_table.append(
            (sample_description_size, data_format, data_reference_index))
        print('%d: size: 0x%X, format: %s, ref_index: 0x%X' % (
            i, sample_description_size, data_format, data_reference_index))

def parse_stsz(buf):
    #Sample Size Atoms

    version   = buf[0]
    flags     = buf[1:4]
    sample_size = struct.unpack('>I', buf[4:8])[0]
    n_entries = struct.unpack('>I', buf[8:12])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'sample_size       : {sample_size}')
    print(f'number of entries : {n_entries}')

    sizes = []
    for i in range(n_entries):
        i0 = 12 + i*4
        i1 = i0 + 4
        if len(buf) < i1: break
        size = struct.unpack('>I', buf[i0:i1])[0]
        sizes.append(size)
        print(f'  {i}: {size}')

def parse_stsc(buf):
    #Sample-to-Chunk Atoms

    version   = buf[0]
    flags     = buf[1:4]
    n_entries = struct.unpack('>I', buf[4:8])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'number of entries : {n_entries}')

    stoc = []
    for i in range(n_entries):
        i0 = 8 + i*12
        i1 = i0 + 12
        if len(buf) < i1: break
        first_chunk       = struct.unpack('>I', buf[i0:i0+4])[0]
        samples_per_chunk = struct.unpack('>I', buf[i0+4:i0+8])[0]
        sample_desc_id    = struct.unpack('>I', buf[i0+8:i0+12])[0]
        stoc.append((first_chunk, samples_per_chunk, sample_desc_id))
        print(f'  {i}: {(first_chunk, samples_per_chunk, sample_desc_id)}')

def parse_stco(buf):
    #Chunk Offset Atoms

    version   = buf[0]
    flags     = buf[1:4]
    n_entries = struct.unpack('>I', buf[4:8])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'number of entries : {n_entries}')

    chunk_offset_table = []
    for i in range(n_entries):
        i0 = 8 + i*4
        i1 = i0 + 4
        if len(buf) < i1: break
        offset = struct.unpack('>I', buf[i0:i1])[0]
        chunk_offset_table.append(offset)
        print(f'  {i}: {offset}')

def parse_co64(buf):
    #64-bit chunk offset atoms

    version   = buf[0]
    flags     = buf[1:4]
    n_entries = struct.unpack('>I', buf[4:8])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'number of entries : {n_entries}')

    chunk_offset_table = []
    for i in range(n_entries):
        i0 = 8 + i*8
        i1 = i0 + 8
        if len(buf) < i1: break
        offset = struct.unpack('>Q', buf[i0:i1])[0]
        chunk_offset_table.append(offset)
        print(f'  {i}: {offset}')

def parse_stts(buf):
    #Time-to-Sample Atoms

    version   = buf[0]
    flags     = buf[1:4]
    n_entries = struct.unpack('>I', buf[4:8])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'number of entries : {n_entries}')

    time_to_sample_table = []
    for i in range(n_entries):
        i0 = 8 + i*8
        i1 = i0 + 8
        if len(buf) < i1: break
        sample_count    = struct.unpack('>I', buf[i0:i0+4])[0]
        sample_duration = struct.unpack('>I', buf[i0+4:i0+8])[0]
        time_to_sample_table.append((sample_count, sample_duration))
        print(f'  {i}: {(sample_count, sample_duration)}')

def parse_stss(buf):
    #Sync Sample Atoms

    version   = buf[0]
    flags     = buf[1:4]
    n_entries = struct.unpack('>I', buf[4:8])[0]

    print(f'version           : {version}')
    print(f'flags             : {flags}')
    print(f'number of entries : {n_entries}')

    sync_sample_table = []
    for i in range(n_entries):
        i0 = 8 + i*4
        i1 = i0 + 4
        if len(buf) < i1: break
        sample = struct.unpack('>I', buf[i0:i1])[0]
        sync_sample_table.append(sample)
        print(f'  {i}: {sample}')

def parse_uuid(buf):
    print_binaries(buf[:16])
    print('%s' % str(buf[16:], 'utf-8'))


def print_binaries(buf, cur=None):
    if cur is None: cur = 0
    for i in range(0, len(buf), 8):
        print('%010X : ' % (i+cur), end='')
        j = min(i+8, len(buf))
        buf_ = buf[i:j]
        print(' '.join(['%02X'%(b) for b in buf_]), end='')
        print(' : ', end='')
        print(''.join(['%c'%(b) for b in buf_]))        


def print_atom_headers(f, verbose=False, pre_label=''):
    atom_start = f.tell()
    buf = f.read(8)

    n = struct.unpack('>I', buf[:4])[0]
    atom_type = str(buf[4:], 'utf-8')

    if n == 1:
        # decode 64-bit size
        buf = f.read(8)
        n = struct.unpack('>Q', buf)[0]
    #elif n == 0:
    #    raise ValueError('not implemented yet')

    #print(f'{atom_type} (size: {n})')
    if not pre_label is None:
        print('%s%s (size: 0x%X)' % (pre_label, atom_type, n))
    else:
        print('%s (size: 0x%X)' % (atom_type, n))
    data_start = f.tell()
    if verbose: print_binaries(buf, atom_start)


    if not atom_type in ('moov', 'trak', 'mdia', 'minf', 'edts', 'dinf', 'stbl'):
        if n > 8:
            if atom_type == 'uuid':
                n_ = n
            else:
                n_ = min(n, 128)

            buf = f.read(n_-8)
            if atom_type == 'mvhd':
                parse_mvhd(buf)
            elif atom_type == 'tkhd':
                parse_tkhd(buf)
            elif atom_type == 'mdhd':
                parse_mdhd(buf)
            elif atom_type == 'stsd':
                parse_stsd(buf)
            elif atom_type == 'stsz':
                parse_stsz(buf)
            elif atom_type == 'stsc':
                parse_stsc(buf)
            elif atom_type == 'stco':
                parse_stco(buf)
            elif atom_type == 'co64':
                parse_co64(buf)
            elif atom_type == 'stts':
                parse_stts(buf)
            elif atom_type == 'stss':
                parse_stss(buf)
            elif atom_type == 'uuid':
                parse_uuid(buf)
            else:
                print('DATA:')
                print_binaries(buf, cur=data_start)
    else:
        # sub Atoms
        sub_end = atom_start + n
        sub_cur = data_start
        while True:
            f.seek(sub_cur)
            if f.tell() != sub_cur: raise ValueError(f'seek failed? {f.tell()} != {sub_cur}')
            sub_n, sub_type = print_atom_headers(f, verbose=False, pre_label=pre_label+atom_type+ ' / ')
            if sub_n == 0: break
            sub_cur += sub_n
            if sub_cur >= sub_end: break
            print('')

    return n, atom_type


def print_atoms(filename, verbose=False):
    with open(filename, 'rb') as f:
        f.seek(0, 2)
        file_size = f.tell()
        print('file size : 0x%010X' % (file_size))
        print('')

        cur = 0
        while True:
            f.seek(cur)
            if f.tell() != cur: raise ValueError(f'seek failed? {f.tell()} != {cur}')
            n, _ = print_atom_headers(f, verbose=verbose)
            print('size : 0x%X' % (n))
            if n == 0: break
            cur += n
            if cur >= file_size: break
            print('')


# ## extracting `moov` as a reference

def read_atom_head(f):
    cur = f.tell()
    buf = f.read(8)

    n = struct.unpack('>I', buf[:4])[0]
    atom_type = str(buf[4:], 'utf-8')

    buf2 = None
    if n == 1:
        # decode 64-bit size
        buf2 = f.read(8)
        n = struct.unpack('>Q', buf2)[0]

    del buf
    del buf2
    gc.collect()

    return n, atom_type


def extract_moov(src_filename, dst_filename, n_chunk=65536, verbose=False):
    with open(src_filename, 'rb') as f_src,        open(dst_filename, 'wb') as f_dst:

        f_src.seek(0, 2)
        src_end = f_src.tell()

        # look for 'moov'
        src_cur = 0
        while True:
            f_src.seek(src_cur)
            if f_src.tell() != src_cur: raise ValueError(f'seek failed? {f_src.tell()} != {src_cur}')

            n, atom_type = read_atom_head(f_src)
            if atom_type == 'moov': break
            src_cur += n

        # 'moov' is found
        moov_start = src_cur

        # copy moov
        f_src.seek(moov_start)
        if f_src.tell() != moov_start: raise ValueError(f'seek failed? {f_src.tell()} != {moov_start}')

        if verbose:
            it_moov = tqdm(range(moov_start, src_end, n_chunk))
        else:
            it_moov = range(moov_start, src_end, n_chunk)
        #for src_cur in tqdm(range(moov_start, src_end, n_chunk)):
        for src_cur in it_moov:
            f_dst.write(f_src.read(n_chunk))
        if src_end - src_cur > 0:
            f_dst.write(f_src.read(src_end - src_cur))


# ## regenerating sample tables from `mdat`

def recover_sample_tables_from_mdat_fast(filename, verbose=False):
    mov_table = []
    aac_table = []

    with open(filename, 'rb') as f_in:

        # look for 'mdat'
        src_cur = 0
        while True:
            f_in.seek(src_cur)
            if f_in.tell() != src_cur: raise ValueError(f'seek failed? {f_in.tell()} != {src_cur}')

            n, atom_type = read_atom_head(f_in)
            if atom_type == 'mdat': break
            src_cur += n

        # 'mdat' is found
        mdat_start = src_cur
        if n == 0:
            # mdat from impcomplete mp4 file
            f_in.seek(0, 2)
            mdat_end = f_in.tell()
            # seek the data_start position
            # 8 bytes for the header PLUS 8 bytes for the reserved space of the size
            f_in.seek(src_cur + 16)
        else:
            mdat_end   = src_cur + n

        n = 0
        while True:
            cur = f_in.tell()
            if cur >= mdat_end: break

            buf = f_in.read(4)

            if buf[0] != 0xFF or buf[1] != 0xF1 or buf[2] != 0x4C or (buf[3] & 0b11111100) != 0x80:
                # h264 chunk
                frame_length = struct.unpack('>I', buf)[0] + 4
                if cur+frame_length >= mdat_end: break

                if verbose: print(f'{n}: [mov] {cur}, {frame_length}')
                mov_table.append((cur, frame_length))
                f_in.seek(cur+frame_length)
            else:
                buf_2 = f_in.read(2)

                # from https://wiki.multimedia.cx/index.php/ADTS
                # AAAAAAAA AAAABCCD EEFFFFGH HHIJKLMM MMMMMMMM MMMOOOOO OOOOOOPP (QQQQQQQQ QQQQQQQQ)
                # 0th-byte 1st      2nd      3rd      4th      5th      6th      (7th      8th     )
                # 0xFF     0xF1     0x4C     0X80 -- typical case for Insta360 ONE-X
                # M 13 frame length, this value must include 7 or 9 bytes of header length
                #   FrameLength = (ProtectionAbsent == 1 ? 7 : 9) + size(AACFrame)
                frame_length = ((buf[3] & 0b11) << 11) | (buf_2[0] << 3) | (buf_2[1] >> 5)
                if cur+frame_length >= mdat_end: break

                if verbose: print(f'{n}: [aac] {cur}, {frame_length}')
                aac_table.append((cur, frame_length))
                f_in.seek(cur+frame_length)

            n += 1

    return mov_table, aac_table


# ## rebuilding `moov` from sample tables

def copy_atom_box(target_type, target_size, f_src, f_dst, only_header=True):
    src_size, atom_type = read_atom_head(f_src)
    if atom_type != target_type: raise ValueError(f'{target_type} not found but {atom_type}')

    if target_size is None: target_size = src_size

    f_dst.write(struct.pack('>I', target_size))
    f_dst.write(target_type.encode('utf-8'))

    if not only_header:
        f_dst.write(f_src.read(target_size-8))

    return src_size


def recover_moov_from_sample_tables(
    ref_filename, dst_filename,
    mov_table, aac_table,
    full_copy=True, n_chunk=65536,
    verbose=False,
    ):

    # constants
    mov_sample_duration = 1001
    aac_sample_duration = 1024

    mvhd_timescale = 48000
    mov_timescale = 30000
    aac_timescale = 48000

    n_mov_table = len(mov_table)
    n_aac_table = len(aac_table)

    mov_mdhd_duration = n_mov_table * mov_sample_duration
    aac_mdhd_duration = n_aac_table * aac_sample_duration
    mov_tkhd_duration = int(mov_mdhd_duration * mvhd_timescale / mov_timescale)


    sample_size_tables = []
    sample_size_tables.append([s for o, s in mov_table])
    sample_size_tables.append([s for o, s in aac_table])

    chunk_offset_tables = []
    chunk_offset_tables.append([o for o, s in mov_table])
    chunk_offset_tables.append([o for o, s in aac_table])


    # moov structure is assumed to be in the fixed format (for now)
    mov_stsz_size = len(sample_size_tables[0])* 4 + 20
    aac_stsz_size = len(sample_size_tables[1])* 4 + 20

    mov_co64_size = len(chunk_offset_tables[0])* 8 + 16
    aac_co64_size = len(chunk_offset_tables[1])* 8 + 16

    mov_stss_size = ((len(sample_size_tables[0])-1)//32 + 1)* 4 + 16


    mov_stbl_size = 8 + 0x141 + 0x18 + 0x1C + mov_stsz_size + mov_co64_size + mov_stss_size
    aac_stbl_size = 8 + 0x82  + 0x18 + 0x1C + aac_stsz_size + aac_co64_size

    mov_minf_size = 8 + 0x14 + 0x24 + mov_stbl_size
    aac_minf_size = 8 + 0x10 + 0x24 + aac_stbl_size

    mov_mdia_size = 8 + 0x20 + 0x2E + mov_minf_size
    aac_mdia_size = 8 + 0x20 + 0x2E + aac_minf_size

    mov_trak_size = 8 + 0x5C + 0x24 + mov_mdia_size + 0x618
    aac_trak_size = 8 + 0x5C + 0x24 + aac_mdia_size

    moov_size = 8 + 0x6C + 0x73 + mov_trak_size + aac_trak_size


    with open(ref_filename, 'rb') as f_moov,        open(dst_filename, 'wb') as f_dst:

        f_moov.seek(0, 2)
        file_size = f_moov.tell()

        cur = 0
        f_moov.seek(cur)
        if f_moov.tell() != cur: raise ValueError(f'seek failed? {f_moov.tell()} != {cur}')

        # moov
        copy_atom_box('moov', moov_size, f_moov, f_dst, only_header=True)

        #copy_atom_box('mvhd', None, f_moov, f_dst, only_header=False)
        # mvhd : duration = mov_tkhd_duration
        n = copy_atom_box('mvhd', None, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        # the following is unchanged
        f_dst.write(buf[:16])
        #duration           = struct.unpack('>I', buf[16:20])[0]
        f_dst.write(struct.pack('>I', mov_tkhd_duration))
        # the rest is unchanged
        f_dst.write(buf[20:])
        #...
        #next_track_id      = struct.unpack('>I', buf[96:100])[0]
        if n != (100+8): raise ValueError(f'ERROR: mov tkhd box size is not 108 but {n}')

        copy_atom_box('udta', None, f_moov, f_dst, only_header=False)

        # movie track
        # trak
        copy_atom_box('trak', mov_trak_size, f_moov, f_dst, only_header=True)

        #copy_atom_box('tkhd', None, f_moov, f_dst, only_header=False)
        # tkhd : duration = mov_tkhd_duration
        n = copy_atom_box('tkhd', None, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        # the following is unchanged
        f_dst.write(buf[:20])
        #duration           = struct.unpack('>I', buf[20:24])[0]
        f_dst.write(struct.pack('>I', mov_tkhd_duration))
        # the rest is unchanged
        f_dst.write(buf[24:])
        #...
        #track_height       = struct.unpack('>I', buf[80:84])[0]
        if n != (84+8): raise ValueError(f'ERROR: mov tkhd box size is not 92 but {n}')

        copy_atom_box('edts', None, f_moov, f_dst, only_header=False)

        # mdia
        copy_atom_box('mdia', mov_mdia_size, f_moov, f_dst, only_header=True)

        #copy_atom_box('mdhd', None, f_moov, f_dst, only_header=False)
        # mdhd : duration = mov_mdhd_duration
        n = copy_atom_box('mdhd', None, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        # the following is unchanged
        f_dst.write(buf[:16])
        #duration           = struct.unpack('>I', buf[16:20])[0]
        f_dst.write(struct.pack('>I', mov_mdhd_duration))
        # the rest is unchanged
        f_dst.write(buf[20:])
        #...
        #quality            = struct.unpack('>H', buf[22:24])[0]
        if n != (24+8): raise ValueError(f'ERROR: mov mdhd box size is not 32 but {n}')

        copy_atom_box('hdlr', None, f_moov, f_dst, only_header=False)

        # minf
        copy_atom_box('minf', mov_minf_size, f_moov, f_dst, only_header=True)
        copy_atom_box('vmhd', None, f_moov, f_dst, only_header=False)
        copy_atom_box('dinf', None, f_moov, f_dst, only_header=False)

        # stbl
        copy_atom_box('stbl', mov_stbl_size, f_moov, f_dst, only_header=True)
        copy_atom_box('stsd', None, f_moov, f_dst, only_header=False)

        #copy_atom_box('stts', None, f_moov, f_dst, only_header=False)
        # stts : sample_count = n_mov_table
        n = copy_atom_box('stts', None, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        f_dst.write(buf[:4]) # version + flags
        f_dst.write(struct.pack('>I', 1)) # n_entries
        f_dst.write(struct.pack('>I', n_mov_table)) # sample_count
        f_dst.write(struct.pack('>I', mov_sample_duration)) # sample_duration

        copy_atom_box('stsc', None, f_moov, f_dst, only_header=False)

        # stsz
        n = copy_atom_box('stsz', mov_stsz_size, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        f_dst.write(buf[:4]) # version + flags
        f_dst.write(struct.pack('>I', 0)) # sample_size
        f_dst.write(struct.pack('>I', len(sample_size_tables[0]))) # n_entries
        for sz in sample_size_tables[0]:
            f_dst.write(struct.pack('>I', sz))

        # co64
        n = copy_atom_box('co64', mov_co64_size, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        f_dst.write(buf[:4]) # version + flags
        f_dst.write(struct.pack('>I', len(chunk_offset_tables[0]))) # n_entries
        for co in chunk_offset_tables[0]:
            f_dst.write(struct.pack('>Q', co))

        # stss
        n = copy_atom_box('stss', mov_stss_size, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        f_dst.write(buf[:4]) # version + flags
        mov_stss_entries = (len(sample_size_tables[0])-1)//32 + 1
        f_dst.write(struct.pack('>I', mov_stss_entries)) # n_entries
        ss = 1
        for i_ss in range(mov_stss_entries):
            f_dst.write(struct.pack('>I', ss))
            ss += 32

        # uuid
        copy_atom_box('uuid', None, f_moov, f_dst, only_header=False)


        # audio track
        # trak
        copy_atom_box('trak', aac_trak_size, f_moov, f_dst, only_header=True)

        #copy_atom_box('tkhd', None, f_moov, f_dst, only_header=False)
        # tkhd : duration = aac_mdhd_duration
        n = copy_atom_box('tkhd', None, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        # the following is unchanged
        f_dst.write(buf[:20])
        #duration           = struct.unpack('>I', buf[20:24])[0]
        f_dst.write(struct.pack('>I', aac_mdhd_duration))
        # the rest is unchanged
        f_dst.write(buf[24:])
        #...
        #track_height       = struct.unpack('>I', buf[80:84])[0]
        if n != (84+8): raise ValueError(f'ERROR: audio tkhd box size is not 92 but {n}')

        copy_atom_box('edts', None, f_moov, f_dst, only_header=False)

        # mdia
        copy_atom_box('mdia', aac_mdia_size, f_moov, f_dst, only_header=True)

        #copy_atom_box('mdhd', None, f_moov, f_dst, only_header=False)
        # mdhd : duration = aac_mdhd_duration
        n = copy_atom_box('mdhd', None, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        # the following is unchanged
        f_dst.write(buf[:16])
        #duration           = struct.unpack('>I', buf[16:20])[0]
        f_dst.write(struct.pack('>I', aac_mdhd_duration))
        # the rest is unchanged
        f_dst.write(buf[20:])
        #...
        #quality            = struct.unpack('>H', buf[22:24])[0]
        if n != (24+8): raise ValueError(f'ERROR: audio mdhd box size is not 32 but {n}')

        copy_atom_box('hdlr', None, f_moov, f_dst, only_header=False)

        # minf
        copy_atom_box('minf', aac_minf_size, f_moov, f_dst, only_header=True)
        copy_atom_box('smhd', None, f_moov, f_dst, only_header=False)
        copy_atom_box('dinf', None, f_moov, f_dst, only_header=False)

        # stbl
        copy_atom_box('stbl', aac_stbl_size, f_moov, f_dst, only_header=True)
        copy_atom_box('stsd', None, f_moov, f_dst, only_header=False)

        #copy_atom_box('stts', None, f_moov, f_dst, only_header=False)
        # stts : sample_count = n_aac_table
        n = copy_atom_box('stts', None, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        f_dst.write(buf[:4]) # version + flags
        f_dst.write(struct.pack('>I', 1)) # n_entries
        f_dst.write(struct.pack('>I', n_aac_table)) # sample_count
        f_dst.write(struct.pack('>I', aac_sample_duration)) # sample_duration

        copy_atom_box('stsc', None, f_moov, f_dst, only_header=False)

        # stsz
        n = copy_atom_box('stsz', aac_stsz_size, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        f_dst.write(buf[:4]) # version + flags
        f_dst.write(struct.pack('>I', 0)) # sample_size
        f_dst.write(struct.pack('>I', len(sample_size_tables[1]))) # n_entries
        for sz in sample_size_tables[1]:
            f_dst.write(struct.pack('>I', sz))

        # co64
        n = copy_atom_box('co64', aac_co64_size, f_moov, f_dst, only_header=True)
        buf = f_moov.read(n-8)
        f_dst.write(buf[:4]) # version + flags
        f_dst.write(struct.pack('>I', len(chunk_offset_tables[1]))) # n_entries
        for co in chunk_offset_tables[1]:
            f_dst.write(struct.pack('>Q', co))

        if not full_copy: return

        # just copy the rest of reference moov file
        moov_cur = f_moov.tell()
        f_moov.seek(0, 2)
        moov_size = f_moov.tell()
        f_moov.seek(moov_cur)
        if f_moov.tell() != moov_cur: raise ValueError(f'seek failed? {f_moov.tell()} != {moov_cur}')
        if verbose:
            it_moov = tqdm(range(moov_cur, moov_size, n_chunk))
        else:
            it_moov = range(moov_cur, moov_size, n_chunk)
        #for moov_cur in tqdm(range(moov_cur, moov_size, n_chunk)):
        for moov_cur in it_moov:
            f_dst.write(f_moov.read(n_chunk))
        if moov_size - moov_cur > 0:
            f_dst.write(f_moov.read(moov_size - moov_cur))


# ## merging the recovered `moov`

def merge_moov(
    src_filename,
    moov_filename,
    dst_filename,
    n_chunk=65536,
    verbose=False):

    with open(src_filename, 'rb') as f_src,        open(moov_filename, 'rb') as f_moov,        open(dst_filename, 'wb') as f_dst:

        f_src.seek(0, 2)
        file_size = f_src.tell()

        cur = 0
        f_src.seek(cur)
        if f_src.tell() != cur: raise ValueError(f'seek failed? {f_src.tell()} != {cur}')

        # ftyp
        n, atom_type = read_atom_head(f_src)
        if atom_type != 'ftyp': raise ValueError('ftyp not found')

        f_src.seek(cur)
        if f_src.tell() != cur: raise ValueError(f'seek failed? {f_src.tell()} != {cur}')
        buf = f_src.read(n)
        f_dst.write(buf)
        if verbose:
            print_binaries(buf)
        cur += n

        # mdat
        n, atom_type = read_atom_head(f_src)
        if atom_type != 'mdat': raise ValueError('mdat not found')

        if n != 0: raise ValueError('size would be zero...')

        # fixed mdat header
        if verbose:
            print_binaries(struct.pack('>Icccc', 1, b'm', b'd', b'a', b't'))
            print_binaries(struct.pack('>Q', file_size-0x20))
        f_dst.write(struct.pack('>Icccc', 1, b'm', b'd', b'a', b't'))
        f_dst.write(struct.pack('>Q', file_size-0x20))

        cur += 16
        f_src.seek(cur)
        if f_src.tell() != cur: raise ValueError(f'seek failed? {f_src.tell()} != {cur}')
        if verbose:
            it_cur = tqdm(range(cur, file_size, n_chunk))
        else:
            it_cur = range(cur, file_size, n_chunk)
        #for cur in tqdm(range(cur, file_size, n_chunk)):
        for cur in it_cur:
            f_dst.write(f_src.read(n_chunk))
        if file_size - cur > 0:
            f_dst.write(f_src.read(file_size - cur))
        print('')

        # search moov
        f_moov.seek(0, 2)
        moov_size = f_moov.tell()
        if verbose:
            print(f'moov_size: {moov_size}')

        moov_cur = 0
        f_moov.seek(moov_cur)
        if f_moov.tell() != moov_cur: raise ValueError(f'seek failed? {f_moov.tell()} != {moov_cur}')
        n, atom_type = read_atom_head(f_moov)
        if atom_type != 'moov': raise ValueError(f'something is wrong...')

        # copy moov
        f_moov.seek(moov_cur)
        if f_moov.tell() != moov_cur: raise ValueError(f'seek failed? {f_moov.tell()} != {moov_cur}')
        if verbose:
            it_moov = tqdm(range(moov_cur, moov_size, n_chunk))
        else:
            it_moov = range(moov_cur, moov_size, n_chunk)
        #for moov_cur in tqdm(range(moov_cur, moov_size, n_chunk)):
        for moov_cur in it_moov:
            f_dst.write(f_moov.read(n_chunk))
        if moov_size - moov_cur > 0:
            f_dst.write(f_moov.read(moov_size - moov_cur))


# # main program to recover corrupted MP4

def finsta360(
    src_filename,
    ref_filename=None,
    dst_filename=None,
    keep_temp=False,
    verbose=False):

    if ref_filename is None:
        # check mode
        print_atoms(src_filename)
        return

    # temporary files
    ref_moov_filename = 'finsta360_ref.moov'
    new_moov_filename = 'finsta360_new.moov'

    # 1) extract reference moov
    print('')
    print('########################################')
    print(f'# 1) extracting reference moov from\n\t{ref_filename}')
    extract_moov(ref_filename, ref_moov_filename)
    if verbose:
        print_atoms(ref_moov_filename)

    # 2) regenerate sample tables from mdat
    print('')
    print('########################################')
    print(f'# 2) regenerate sample tables from mdat in\n\t{src_filename}')
    mov_table, aac_table = recover_sample_tables_from_mdat_fast(
        src_filename,
        verbose=False)
    if verbose:
        print(f'number of samples (movie) : {len(mov_table)}')
        print(f'number of samples (audio) : {len(aac_table)}')
        # constants
        mov_sample_duration = 1001
        aac_sample_duration = 1024

        mvhd_timescale = 48000
        mov_timescale = 30000
        aac_timescale = 48000

        n_mov_table = len(mov_table)
        n_aac_table = len(aac_table)

        mov_mdhd_duration = n_mov_table * mov_sample_duration
        aac_mdhd_duration = n_aac_table * aac_sample_duration
        mov_tkhd_duration = int(mov_mdhd_duration * mvhd_timescale / mov_timescale)

        # mvhd
        mvhd_duration_sec = mov_tkhd_duration / mvhd_timescale
        print(f'mvhd duration  : {mvhd_duration_sec} sec / {mvhd_duration_sec/60} min')
        # movie mdhd
        mov_duration_sec = mov_mdhd_duration / mov_timescale
        print(f'movie duration : {mov_duration_sec} sec / {mov_duration_sec/60} min')
        # audio mdhd
        aac_duration_sec = aac_mdhd_duration / aac_timescale
        print(f'audio duration : {aac_duration_sec} sec / {aac_duration_sec/60} min')

    # 3) rebuilding moov from the sample tables
    print('')
    print('########################################')
    print(f'# 3) rebuilding moov from the sample tables')
    recover_moov_from_sample_tables(
        ref_moov_filename,
        new_moov_filename,
        mov_table, aac_table,
        full_copy=True,
    )
    if verbose:
        print_atoms(new_moov_filename)

    if dst_filename is None:
        # test mode
        if not keep_temp:
            os.remove(ref_moov_filename)
            os.remove(new_moov_filename)
        return

    # 4) merging the rebuilt moov into the source
    print('')
    print('########################################')
    print(f'# 4) merging the rebuilt moov into\n\t{src_filename}\nas\n\t{dst_filename}')
    merge_moov(
        src_filename,
        new_moov_filename,
        dst_filename,
    )


    if not keep_temp:
        os.remove(ref_moov_filename)
        os.remove(new_moov_filename)


def usage():
    print('finsta360.py : to finalize incomplete MP4 of Insta360 ONE-X
    print('https://github.com/kichiki/finsta360')
    print('USAGE: finsta360.py [options]')
    print('\t-s file : source file, that is, corrupted mp4 (insv) file')
    print('\t-r file : complete mp4 (insv) file as a reference')
    print('\t-o file : output recovered mp4 (insv) file')
    print('\t-v      : to set verbose mode')
    print('\t-k      : to keep temporary files')
    print('\t          (reference and recovered moov files, finsta360*.moov)')
    print('If you provide only source file (-s), program prints the metadata')
    print('If you dont provide output file (-o), program just runs without writing')
    sys.exit ()


if __name__ == '__main__':
    src_filename = None
    ref_filename = None
    dst_filename = None
    verbose = False
    keep_temp = False
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '-s':
            src_filename = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '-r':
            ref_filename = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '-o':
            dst_filename = sys.argv[i+1]
            i += 2
        elif sys.argv[i] == '-v':
            verbose = True
            i += 1
        elif sys.argv[i] == '-k':
            keep_temp = True
            i += 1
        else:
            usage()
            break

    if src_filename is None:
        print(f'you must provie source file {src_filename}')
        usage()
        sys.exit()
    if not os.path.exists(src_filename):
        print(f'source file {src_filename} does not exist')
        sys.exit()
    if not ref_filename is None and not os.path.exists(ref_filename):
        print(f'reference file {ref_filename} does not exist')
        sys.exit()
    if not dst_filename is None and os.path.exists(dst_filename):
        print(f'output file {dst_filename} already exists')
        sys.exit()


    finsta360(
        src_filename,
        ref_filename,
        dst_filename,
        keep_temp,
        verbose)


    sys.exit()
