import ipaddress
import random
from warnings import warn

from exrex import getone

MAX_IPV4 = ipaddress.IPv4Address._ALL_ONES  # 2 ** 32 - 1
MAX_IPV6 = ipaddress.IPv6Address._ALL_ONES  # 2 ** 128 - 1

MAX_ITER = 100_000


def random_ipv4():
    return ipaddress.IPv4Address._string_from_ip_int(random.randint(0, MAX_IPV4))


def random_ipv6():
    return ipaddress.IPv6Address._string_from_ip_int(random.randint(0, MAX_IPV6))


def get_one_regex(regex):
    return lambda: getone(regex)


def get_n_unique_from_one(gen, n, max_iter: int | None = None, unique: bool = True):
    if max_iter is None:
        max_iter = max(3 * n, MAX_ITER)
    res = set()
    it = 0
    while len(res) != n and it < max_iter:
        val = gen()
        res.add(val)
        it += 1
    if it >= max_iter:
        if unique:
            warn("Could not generate enough unique values with the anonymizer!")
        res = list(res)
        missing = n - len(res)
        for i in range(missing):
            res.append(gen())
    return list(res)
