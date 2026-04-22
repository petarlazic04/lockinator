import requests
from requests.auth import HTTPDigestAuth


class Camera:
    def __init__(self, ip, port=80, user="admin", password="admin", debug=False):
        self.base_url = f"http://{ip}:{port}"
        self.auth = HTTPDigestAuth(user, password)
        self.debug = debug
        
    def _soap(self, endpoint, body_xml):
        envelope = f"""<?xml version="1.0" encoding="UTF-8"?>
<s:Envelope
  xmlns:s="http://www.w3.org/2003/05/soap-envelope"
  xmlns:tptz="http://www.onvif.org/ver20/ptz/wsdl"
  xmlns:tt="http://www.onvif.org/ver10/schema">
  <s:Header/>
  <s:Body>
    {body_xml}
  </s:Body>
</s:Envelope>"""

        url = f"{self.base_url}{endpoint}"

        try:
            resp = requests.post(
                url,
                data=envelope.encode("utf-8"),
                headers={"Content-Type": "application/soap+xml; charset=utf-8"},
                auth=self.auth,
                timeout=5
            )

            if self.debug:
                print(f"[DEBUG] {endpoint} → {resp.status_code}")

            return resp

        except requests.exceptions.RequestException as e:
            print(f"[!] Error: {e}")
            return None

    def move(self, pan=0.0, tilt=0.0, zoom=0.0, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:ContinuousMove>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
          <tptz:Velocity>
            <tt:PanTilt x="{pan:.4f}" y="{tilt:.4f}"/>
            <tt:Zoom x="{zoom:.4f}"/>
          </tptz:Velocity>
        </tptz:ContinuousMove>
        """)

    def stop(self, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:Stop>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
          <tptz:PanTilt>true</tptz:PanTilt>
          <tptz:Zoom>true</tptz:Zoom>
        </tptz:Stop>
        """)

    def absolute_move(self, pan=0.0, tilt=0.0, zoom=0.0, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:AbsoluteMove>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
          <tptz:Position>
            <tt:PanTilt x="{pan:.4f}" y="{tilt:.4f}"/>
            <tt:Zoom x="{zoom:.4f}"/>
          </tptz:Position>
        </tptz:AbsoluteMove>
        """)

    def relative_move(self, pan=0.0, tilt=0.0, zoom=0.0, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:RelativeMove>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
          <tptz:Translation>
            <tt:PanTilt x="{pan:.4f}" y="{tilt:.4f}"/>
            <tt:Zoom x="{zoom:.4f}"/>
          </tptz:Translation>
        </tptz:RelativeMove>
        """)

    def goto_preset(self, token, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:GotoPreset>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
          <tptz:PresetToken>{token}</tptz:PresetToken>
        </tptz:GotoPreset>
        """)

    def set_preset(self, name, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:SetPreset>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
          <tptz:PresetName>{name}</tptz:PresetName>
        </tptz:SetPreset>
        """)

    def get_presets(self, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:GetPresets>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
        </tptz:GetPresets>
        """)

    def get_status(self, profile="Profile_1"):
        return self._soap("/onvif/PTZ", f"""
        <tptz:GetStatus>
          <tptz:ProfileToken>{profile}</tptz:ProfileToken>
        </tptz:GetStatus>
        """)

    def get_profiles(self):
        return self._soap("/onvif/device_service", """
        <tds:GetProfiles xmlns:tds="http://www.onvif.org/ver10/media/wsdl"/>
        """)