// SPDX-License-Identifier: GPL-2.0
/*
 * RB3 Gen 2 CAM0B bring-up without a device-tree change (test only).
 *
 * - binds i2c-qcom-cci to the disabled cci1 node (/soc@0/cci@ac4b000), which
 *   gives /dev/i2c-N for CCI_I2C2 (CAM0B) and CCI_I2C3
 * - powers the CAM0B sensor with the resources from the vendor CamX node
 *   qcom,cam-sensor7: VIO = cam_vio-supply, MCLK4 (24 MHz on GPIO68),
 *   reset = 2nd entry of "gpios" (GPIO149)
 *
 * rmmod undoes everything. Nothing is written to flash.
 */
#include <linux/clk.h>
#include <linux/delay.h>
#include <linux/gpio/consumer.h>
#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_address.h>
#include <linux/pinctrl/consumer.h>
#include <linux/pinctrl/machine.h>
#include <linux/platform_device.h>
#include <linux/regulator/consumer.h>

#define CCI1_PATH	"/soc@0/cci@ac4b000"
#define SENSOR_PATH	"/soc@0/qcom,cci1/qcom,cam-sensor7"
#define BRIDGE_NAME	"cam0b-bridge"

static const struct pinctrl_map mclk_map[] = {
	PIN_MAP_MUX_GROUP(BRIDGE_NAME, "mclk", "f100000.pinctrl", "gpio68", "cam_mclk"),
};

static struct platform_device *cci_pdev, *cam_pdev;
static struct regulator *vio;
static struct clk *mclk;
static struct pinctrl *pins;
static struct gpio_desc *reset;

static struct platform_device *add_of_pdev(const char *name, const char *path, bool mem)
{
	struct platform_device *pdev;
	struct device_node *np;
	struct resource res;
	int ret;

	np = of_find_node_by_path(path);
	if (!np)
		return ERR_PTR(-ENODEV);

	pdev = platform_device_alloc(name, PLATFORM_DEVID_NONE);
	if (!pdev) {
		of_node_put(np);
		return ERR_PTR(-ENOMEM);
	}
	device_set_node(&pdev->dev, of_fwnode_handle(np));	/* keeps the np reference */

	if (mem) {
		ret = of_address_to_resource(np, 0, &res);
		if (!ret)
			ret = platform_device_add_resources(pdev, &res, 1);
		if (ret)
			goto err;
	}

	ret = platform_device_add(pdev);
	if (ret)
		goto err;
	return pdev;
err:
	platform_device_put(pdev);
	return ERR_PTR(ret);
}

static void cam_power_off(void)
{
	if (!IS_ERR_OR_NULL(reset)) {
		gpiod_set_value_cansleep(reset, 0);
		gpiod_put(reset);
	}
	if (!IS_ERR_OR_NULL(mclk)) {
		clk_disable_unprepare(mclk);
		clk_put(mclk);
	}
	if (!IS_ERR_OR_NULL(pins))
		pinctrl_put(pins);
	if (!IS_ERR_OR_NULL(vio)) {
		regulator_disable(vio);
		regulator_put(vio);
	}
}

static int __init cam0b_init(void)
{
	struct device_node *np;
	struct clk *cci_clk;
	struct device *dev;
	int ret;

	/* CCI core clock: 37.5 MHz like cci0, before the driver reads it in probe */
	np = of_find_node_by_path(CCI1_PATH);
	if (!np)
		return -ENODEV;
	cci_clk = of_clk_get_by_name(np, "cci");
	of_node_put(np);
	if (!IS_ERR(cci_clk)) {
		clk_set_rate(cci_clk, 37500000);
		clk_put(cci_clk);
	}

	cci_pdev = add_of_pdev("ac4b000.cci", CCI1_PATH, true);
	if (IS_ERR(cci_pdev))
		return PTR_ERR(cci_pdev);

	ret = pinctrl_register_mappings(mclk_map, ARRAY_SIZE(mclk_map));
	if (ret)
		goto err_cci;

	cam_pdev = add_of_pdev(BRIDGE_NAME, SENSOR_PATH, false);
	if (IS_ERR(cam_pdev)) {
		ret = PTR_ERR(cam_pdev);
		goto err_map;
	}
	dev = &cam_pdev->dev;

	vio = regulator_get(dev, "cam_vio");
	if (IS_ERR(vio)) {
		ret = PTR_ERR(vio);
		goto err_cam;
	}
	ret = regulator_enable(vio);
	if (ret) {
		regulator_put(vio);
		vio = NULL;
		goto err_cam;
	}

	pins = pinctrl_get(dev);
	if (IS_ERR(pins)) {
		ret = PTR_ERR(pins);
		goto err_power;
	}
	ret = pinctrl_select_state(pins, pinctrl_lookup_state(pins, "mclk"));
	if (ret)
		goto err_power;

	mclk = clk_get(dev, "cam_clk");
	if (IS_ERR(mclk)) {
		ret = PTR_ERR(mclk);
		goto err_power;
	}
	clk_set_rate(mclk, 24000000);
	ret = clk_prepare_enable(mclk);
	if (ret) {
		clk_put(mclk);
		mclk = NULL;
		goto err_power;
	}

	reset = gpiod_get_index(dev, NULL, 1, GPIOD_OUT_LOW);
	if (IS_ERR(reset)) {
		ret = PTR_ERR(reset);
		goto err_power;
	}
	usleep_range(1000, 2000);
	gpiod_set_value_cansleep(reset, 1);
	usleep_range(10000, 12000);

	dev_info(dev, "CAM0B powered: MCLK %lu Hz, reset gpio %d high\n",
		 clk_get_rate(mclk), desc_to_gpio(reset));
	return 0;

err_power:
	cam_power_off();
err_cam:
	platform_device_unregister(cam_pdev);
err_map:
	pinctrl_unregister_mappings(mclk_map);
err_cci:
	platform_device_unregister(cci_pdev);
	return ret;
}

static void __exit cam0b_exit(void)
{
	cam_power_off();
	platform_device_unregister(cam_pdev);
	pinctrl_unregister_mappings(mclk_map);
	platform_device_unregister(cci_pdev);
}

module_init(cam0b_init);
module_exit(cam0b_exit);
MODULE_DESCRIPTION("RB3 Gen 2 CAM0B runtime bring-up (cci1 + sensor power) for testing");
MODULE_LICENSE("GPL");
