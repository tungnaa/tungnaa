from typing import Dict, Any, Union
import time

import pythonosc.udp_client as udp_client
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.osc_bundle_builder import OscBundleBuilder

class Sender:
    pass

class SCSynthDirectOSCSender(Sender):
    """sends timed bundles directly to scsynth to set a control bus"""
    def __init__(self, 
            host:str='127.0.0.1', 
            port:int=57110, 
            bus_index:int=64,
            latency:float=0.2,
            ):
        self.client = udp_client.SimpleUDPClient(host, port)
        self.bus_index = bus_index
        self.latency = latency

    def __call__(self, d:Dict[str,Any]):
        """
        Args:
            d: dictionary with 'timestamp' and 'audio_t' keys,
                other keys are ignored in this case
        """
        bb = OscBundleBuilder(d['timestamp'] + self.latency)
        mb = OscMessageBuilder('/c_set')

        latents = [t.float().item() for t in d['latent_t'].squeeze()]
        for i,l in enumerate(latents):
            mb.add_arg(i+self.bus_index)
            mb.add_arg(l)

        bb.add_content(mb.build())
        bundle = bb.build()

        self.client.send(bundle)

        
class GenericOSCSender(Sender):
    def __init__(self, 
            host:str='127.0.0.1', 
            port:int=57120, 
            lroute:str='/tungnaa/latents',
            sroute:str='/tungnaa/status'
            ):
        self.client = udp_client.SimpleUDPClient(host, port)
        self.latents_route = lroute
        self.status_route = sroute

    def __call__(self, d:Dict[str,Any]):
        """
        Args:
            d: dictionary with 'audio_t' key, other keys are ignored in this case

            this should work as a callback for the backend:
            the input dict should be returned to later be enqueued
        """
        latents = [t.float().item() for t in d['latent_t'].squeeze()]
        self.client.send_message(self.latents_route, latents)
        return d

    def send_status(self, cmd:str, value:Union[str,float]):
        """
        Send a non-data message to SuperCollider

        cmd     the status type/command <mute|maybe more...>
        value   the value of the command, e.g. 1.0/0.0 for mute on/off
        """
        self.client.send_message(self.status_route, value)