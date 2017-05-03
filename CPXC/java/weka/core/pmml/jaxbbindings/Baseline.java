//
// This file was generated by the JavaTM Architecture for XML Binding(JAXB) Reference Implementation, v2.0-b52-fcs 
// See <a href="http://java.sun.com/xml/jaxb">http://java.sun.com/xml/jaxb</a> 
// Any modifications to this file will be lost upon recompilation of the source schema. 
// Generated on: 2013.12.20 at 12:48:21 PM GMT 
//


package weka.core.pmml.jaxbbindings;

import java.util.ArrayList;
import java.util.List;

import javax.xml.bind.annotation.XmlAccessType;
import javax.xml.bind.annotation.XmlAccessorType;
import javax.xml.bind.annotation.XmlElement;
import javax.xml.bind.annotation.XmlRootElement;
import javax.xml.bind.annotation.XmlType;


/**
 * <p>Java class for Baseline element declaration.
 * 
 * <p>The following schema fragment specifies the expected content contained within this class.
 * 
 * <pre>
 * &lt;element name="Baseline">
 *   &lt;complexType>
 *     &lt;complexContent>
 *       &lt;restriction base="{http://www.w3.org/2001/XMLSchema}anyType">
 *         &lt;choice>
 *           &lt;group ref="{http://www.dmg.org/PMML-4_1}CONTINUOUS-DISTRIBUTION-TYPES"/>
 *           &lt;group ref="{http://www.dmg.org/PMML-4_1}DISCRETE-DISTRIBUTION-TYPES"/>
 *         &lt;/choice>
 *       &lt;/restriction>
 *     &lt;/complexContent>
 *   &lt;/complexType>
 * &lt;/element>
 * </pre>
 * 
 * 
 */
@XmlAccessorType(XmlAccessType.FIELD)
@XmlType(name = "", propOrder = {
    "anyDistribution",
    "gaussianDistribution",
    "poissonDistribution",
    "uniformDistribution",
    "extension",
    "countTable",
    "normalizedCountTable",
    "fieldRef"
})
@XmlRootElement(name = "Baseline")
public class Baseline {

    @XmlElement(name = "AnyDistribution", namespace = "http://www.dmg.org/PMML-4_1")
    protected AnyDistribution anyDistribution;
    @XmlElement(name = "GaussianDistribution", namespace = "http://www.dmg.org/PMML-4_1")
    protected GaussianDistribution gaussianDistribution;
    @XmlElement(name = "PoissonDistribution", namespace = "http://www.dmg.org/PMML-4_1")
    protected PoissonDistribution poissonDistribution;
    @XmlElement(name = "UniformDistribution", namespace = "http://www.dmg.org/PMML-4_1")
    protected UniformDistribution uniformDistribution;
    @XmlElement(name = "Extension", namespace = "http://www.dmg.org/PMML-4_1", required = true)
    protected List<Extension> extension;
    @XmlElement(name = "CountTable", namespace = "http://www.dmg.org/PMML-4_1")
    protected COUNTTABLETYPE countTable;
    @XmlElement(name = "NormalizedCountTable", namespace = "http://www.dmg.org/PMML-4_1")
    protected COUNTTABLETYPE normalizedCountTable;
    @XmlElement(name = "FieldRef", namespace = "http://www.dmg.org/PMML-4_1", required = true)
    protected List<FieldRef> fieldRef;

    /**
     * Gets the value of the anyDistribution property.
     * 
     * @return
     *     possible object is
     *     {@link AnyDistribution }
     *     
     */
    public AnyDistribution getAnyDistribution() {
        return anyDistribution;
    }

    /**
     * Sets the value of the anyDistribution property.
     * 
     * @param value
     *     allowed object is
     *     {@link AnyDistribution }
     *     
     */
    public void setAnyDistribution(AnyDistribution value) {
        this.anyDistribution = value;
    }

    /**
     * Gets the value of the gaussianDistribution property.
     * 
     * @return
     *     possible object is
     *     {@link GaussianDistribution }
     *     
     */
    public GaussianDistribution getGaussianDistribution() {
        return gaussianDistribution;
    }

    /**
     * Sets the value of the gaussianDistribution property.
     * 
     * @param value
     *     allowed object is
     *     {@link GaussianDistribution }
     *     
     */
    public void setGaussianDistribution(GaussianDistribution value) {
        this.gaussianDistribution = value;
    }

    /**
     * Gets the value of the poissonDistribution property.
     * 
     * @return
     *     possible object is
     *     {@link PoissonDistribution }
     *     
     */
    public PoissonDistribution getPoissonDistribution() {
        return poissonDistribution;
    }

    /**
     * Sets the value of the poissonDistribution property.
     * 
     * @param value
     *     allowed object is
     *     {@link PoissonDistribution }
     *     
     */
    public void setPoissonDistribution(PoissonDistribution value) {
        this.poissonDistribution = value;
    }

    /**
     * Gets the value of the uniformDistribution property.
     * 
     * @return
     *     possible object is
     *     {@link UniformDistribution }
     *     
     */
    public UniformDistribution getUniformDistribution() {
        return uniformDistribution;
    }

    /**
     * Sets the value of the uniformDistribution property.
     * 
     * @param value
     *     allowed object is
     *     {@link UniformDistribution }
     *     
     */
    public void setUniformDistribution(UniformDistribution value) {
        this.uniformDistribution = value;
    }

    /**
     * Gets the value of the extension property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the extension property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getExtension().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link Extension }
     * 
     * 
     */
    public List<Extension> getExtension() {
        if (extension == null) {
            extension = new ArrayList<Extension>();
        }
        return this.extension;
    }

    /**
     * Gets the value of the countTable property.
     * 
     * @return
     *     possible object is
     *     {@link COUNTTABLETYPE }
     *     
     */
    public COUNTTABLETYPE getCountTable() {
        return countTable;
    }

    /**
     * Sets the value of the countTable property.
     * 
     * @param value
     *     allowed object is
     *     {@link COUNTTABLETYPE }
     *     
     */
    public void setCountTable(COUNTTABLETYPE value) {
        this.countTable = value;
    }

    /**
     * Gets the value of the normalizedCountTable property.
     * 
     * @return
     *     possible object is
     *     {@link COUNTTABLETYPE }
     *     
     */
    public COUNTTABLETYPE getNormalizedCountTable() {
        return normalizedCountTable;
    }

    /**
     * Sets the value of the normalizedCountTable property.
     * 
     * @param value
     *     allowed object is
     *     {@link COUNTTABLETYPE }
     *     
     */
    public void setNormalizedCountTable(COUNTTABLETYPE value) {
        this.normalizedCountTable = value;
    }

    /**
     * Gets the value of the fieldRef property.
     * 
     * <p>
     * This accessor method returns a reference to the live list,
     * not a snapshot. Therefore any modification you make to the
     * returned list will be present inside the JAXB object.
     * This is why there is not a <CODE>set</CODE> method for the fieldRef property.
     * 
     * <p>
     * For example, to add a new item, do as follows:
     * <pre>
     *    getFieldRef().add(newItem);
     * </pre>
     * 
     * 
     * <p>
     * Objects of the following type(s) are allowed in the list
     * {@link FieldRef }
     * 
     * 
     */
    public List<FieldRef> getFieldRef() {
        if (fieldRef == null) {
            fieldRef = new ArrayList<FieldRef>();
        }
        return this.fieldRef;
    }

}
